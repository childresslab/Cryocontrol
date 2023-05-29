import numpy as np
import logging
import sys
import numpy.typing as npt

from jpe_coord_convert import JPECoord
from time import time
from pathlib import Path
from subprocess import Popen, run, DEVNULL, CREATE_NEW_CONSOLE
from warnings import warn
from threading import Thread
from typing import Union, Callable, Tuple
from time import sleep

# Create root logger
log = logging.getLogger("stepper")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s - %(name)s:\n\t%(message)s")
# Stream for writing all logged messages to console
ch = logging.StreamHandler(stream=sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
log.addHandler(ch)

# Steam for writing all errors to a file
fh = logging.FileHandler('cryo_errors.txt')
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s:\n\t%(message)s" )
fh.setLevel(logging.WARNING)
fh.setFormatter(formatter)
log.addHandler(fh)

# Default Configuration
# Note how we define z_min is different than jpe document
# since we care about the sign of the position in the jpe coordinates.
# So here, negative value is actually highest mirror position.
stp_config = {"z_min" : -5750.0,            # um
              "z_max" : -250.0,             # um
              "um_per_click" : 5/17,        # um/Click (250um per rev, 850 clicks per rev.)
              "R": 26.0,                    # Actuator Radius - Stage Dependent (mm)
              "h": 58.9,                    # Sample Pedestal Height - Stage Dependent(mm)
              "gain" : 200.0,               # Freq/Error i.e. Hz/Tick
              "stuck_iterations" : 100,     # Integer
              "T" : 10,                     # K
              "z_res" : 0.3,                # um/click
              "xy_res" : 0.8,               # um/click
              "z_rel_lim" : 1.0,            # um
              "z_lim" : [-4520,-3560],      # um
              "xy_rel_lim" : 5.0,           # um
              "xy_lim" : [-1000.0,1000.0],  # um
              "type" : "CA1801",            # Basically never change this
              "serial" : "1038E201702-004", # Basically never change this
              "exe_path" : r"X:\DiamondCloud\Fiber_proj_ctrl_softwares\Cryo Control\JPE vis\CPS_control\cacli.exe",
              "pos_path" : r"X:\DiamondCloud\Fiber_proj_ctrl_softwares\Cryo Control\JPE vis\PositionZeroRegister.asc"} 

stp_conv = JPECoord(stp_config['R'], stp_config['h'],
                    stp_config['z_min'], stp_config['z_max'])

# Error class for indicating bad positioning.
class JPEPositionError(Exception):
    pass

class JPELimitError(Exception):
    pass

class JPEStepper():

    def __init__(self, config:dict[str,any] = stp_config ) -> None:
        f"""The initialization function for a JPE Stepper object.

        Parameters
        ----------
        config : dict[str,any], a dictionary containing the configuration
            of the stage control system. Optional, will default to:
            {stp_config}. If a different dictionary is passed in, this will
            be used to update the default stp_config configuration, allowing
            for only a few configuration variables to be passed in.
        """
        new_config = stp_config.copy()
        new_config.update(stp_config)
        config = new_config
        # These set the absolute maximum positions of the stage
        # Unfortunately we can't easily zero the stage, so these
        # aren't super useful since we have no idea where the stage
        # currently is set
        # Update:
        # We've reset the stage position so that 0,0,0 corresponds to the
        # bottom endstops of the JPE stage.
        # So now, these values actually correspond to how far
        # any one actuator can go.
        self.stp_min = config['z_min']
        self.stp_max = config['z_max']

        # How many microns a single click of the actuator
        # encoder corresponds to, this is in the 'z' basis
        self.um_conv = config['um_per_click']

        # The parameters needed to initialize the stage, be sure
        # to set these depending on the temperature of the cryo!
        self.gain = config['gain']
        self.type = config['type']
        self.temp = config['T']

        # How far the stage can move from it's current position,
        # along each axis, in one motion step.
        # For both limits, a tolerance of resolution/2 is allowed.
        self._rel_lims = np.array([config['xy_rel_lim'],
                                 config['xy_rel_lim'],
                                 config['z_rel_lim']])
        # The lower and upper bounds of absolute motion for each axis.
        # For both limits, a tolerance of resolution/2 is allowed.
        self._lims = np.array([config['xy_lim'],
                             config['xy_lim'],
                             config['z_lim']])
        # Resolution on each axis
        self.res = np.array([config['xy_res'],
                             config['xy_res'],
                             config['z_res']])

        #
        self.stuck_iterations = config["stuck_iterations"]
        
        # The paths to relevant files, namely the executable that calls stage
        # commands, and where we save the position info between instances
        # of the code running.
        self.exe = config['exe_path']
        self.pos_file = Path(config['pos_path'])
        if "emulator" in self.exe:
            self.emulator = True
        else:
            self.emulator = False
            self.pipe = None
        self.serial = config['serial']

        # The vectors that store the relevant zeroing and position offsets
        # As well as wether to subtract the offset for a given axis.
        # The reported position (UserPosition) is computed as: 
        #   StagePosition + OffsetPosition - (Zeroing * ZeroPosition)
        # Where: StagePosition is the position calculated from the clicks
        #        reported by the stage controller software 'cacli'
        #        
        #        OffsetClicks is set in cases where the software position and 
        #        controller position didn't match at initialization to account
        #        for the discrepency. e.g. we know we where at (1325,1325,1325)
        #        and now cacli reports we were at (0,0,0) indicating that
        #        the controller lost power, so we set offset_clicks to 
        #        (1325,1325,1325).
        # 
        #        ZeroPosition is the vector a user can safe to perform offset
        #        position measurements, say we move to (5.0,5.0,5.0) and want
        #        that to now be refered to as (0.0,0.0,5.0). We simply set
        #        the ZeroPosition to be (5.0,5.0,0) and indicate that we're using
        #        xy zeroing.
        self.offset_clicks = np.array([0,0,0])
        self.user_set_position = np.array([0.0,0.0,0.0])
        self.zero_position = np.zeros(3)
        self.zeroing = [False,False]

        # Flag for tracking initialization.
        self.initialized = False
        self.error_flag = False

    def __del__(self) -> None:
        if self.initialized:
            self.deinitialize()
    
    def clicks_to_microns(self, clicks:Union[int,npt.NDArray[np.int]]) -> Union[float, npt.NDArray[np.float64]]:
        """Converts the given number of clicks into microns.

        Parameters
        ----------
        clicks : Union[int,npt.NDArray[int]]
            The amount of clicks to be converted to microns.
            Either a single number or a list of numbers to convert.

        Returns
        -------
        Union[float, npt.NDArray[np.float64]]
            If a single number is input, then return the single number of microns.
            Else, returns an array of the converted values.
        """
        # Handle numpy arrays
        if isinstance(clicks,np.ndarray):
            return (clicks * self.um_conv).astype(float)

        return float(clicks * self.um_conv)

    def microns_to_clicks(self, microns:Union[float,npt.NDArray[np.float64]]) -> Union[int, npt.NDArray[np.int]]:
        """Converts the given number of microns into clicks, including rounding
           to an integer.

        Parameters
        ----------
        microns : Union[float,npt.NDArray[np.float64]]
            The amount of microns to be converted to clicks.
            Either a single number or a list of numbers to convert.

        Returns
        -------
        Union[int, npt.NDArray[int]]
            If a single number is input, then return the single number of clicks.
            Else, returns an array of the converted values.
        """
        # Handle numpy arrays
        if isinstance(microns, np.ndarray):
            return np.round(microns/self.um_conv).astype(np.int)

        return int(round(microns/self.um_conv))

    # User coordinates include position matching offset and toggleable zeroing
    def user_to_stage(self, user_position:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Converts a position in the easier to use 'user' reference frame
           into the more concrete 'stage' reference frame to which the actuator
           clicks actually correspond to.

        Parameters
        ----------
        user_position : np.ndarray[3,float]
            The (x,y,z) position in the user reference frame to be converted.

        Returns
        -------
        np.ndarray[3,float]
            The (x,y,z) position in the stage reference frame.
        """
        # THIS COPY MUST ABSOLUTELY BE DONE
        # OTHERWISE IT WILL OVERRIGHT THE PASSED IN VECTOR
        # CAUSING HORRIBLY COMPLICATED BUGS TO OCCUR
        stage_position = np.copy(user_position)

        zeroing = np.array([self.zeroing[0],self.zeroing[0],self.zeroing[1]], dtype=int)
        stage_position += zeroing * self.zero_position


        return stage_position

    # Stage coordinates is the raw clicks to z position conversion with no convenience
    # offsets.
    def stage_to_user(self, stage_position:npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Converts a position in the concrete 'stage' reference frame into the
        easier to use 'user' reference frame.

        Parameters
        ----------
        stage_position : np.ndarray[3,float]
            The (x,y,z) position in the stage reference frame to be converted.

        Returns
        -------
        np.ndarray[3,float]
            The (x,y,z) position in the user reference frame.
        """
        # THIS COPY MUST ABSOLUTELY BE DONE
        # OTHERWISE IT WILL OVERRIGHT THE PASSESD IN VECTOR
        # CAUSING HORRIBLY COMPLICATED BUGS TO OCCUR
        user_position = np.copy(stage_position)

        zeroing = np.array([self.zeroing[0],self.zeroing[0],self.zeroing[1]], dtype=int)
        user_position -= zeroing * self.zero_position

        return user_position

    def initialize(self) -> None:
        """Initializes the stage. This needs to be done before any action can be
        performed. First the connection to the stage is initialized, and then
        the current reported position is compared to that saved in file. If they
        agree, no adjustments are made, but if they don't we keep track of the
        offset between where the stage should be and where it thinks it is, and
        use this to keep track of the stage_position.
        """
        if not self.emulator:
            self.open_pipe()
        # Reset Click Offset to avoid carrying over
        self.offset_clicks = np.zeros(3,dtype=np.int)
        # Initialize CACLI with current parameters
        self.cacli("initialize", self.gain, self.type, self.temp)
        # Indicate success
        self.initialized = True
        # Check consistency between read and saved position
        # and update accordingly.
        user_pos, zero_pos, zeroing = self.read_pos_file()
        self.zero_position = zero_pos
        self.zeroing = zeroing
        cur_clicks = self.clicks
        cur_pos = self.position
        old_pos = self.user_to_stage(user_pos)
        log.debug(f"Previous position = {old_pos}.")
        log.debug(f"Current Reported Position (No Offset) = {cur_pos}.")
        # Check if all values are within tolerance (~1pm)
        if not np.all(np.abs((cur_pos - old_pos)) < 1E-6):
            # If not, assume the controller has been reset and the saved
            # position is accurate, so set the offset clicks to be the saved position
            old_clicks = self.microns_to_clicks(stp_conv.zs_from_cart(old_pos))
            log.debug(f"Previous Clicks = {old_clicks}")
            log.debug(f"Current Clicks (No Offset) = {cur_clicks}")
            self.offset_clicks = old_clicks-cur_clicks
            self.user_set_position = user_pos
            if np.all(self.offset_clicks==0):
                return
            else:
                log.warn(f"Saved Position discrepancy found, setting {self.offset_clicks = }.")
            if np.all(cur_clicks == 0):
                log.warn(f"All current clicks are zero, was the controller reset?")
        # If everything matches, we don't need to change anything.
        else:
            # Redundent setting to zero, but reads nicely
            self.offset_clicks = np.zeros(3,dtype=np.int)
            self.user_set_position = user_pos
        self.error_flag = False


    def deinitialize(self) -> None:
        """Safely close the stage connection and save the current position to
        a file.
        """
        # Save position to file
        self.write_pos_file()
        # Deinitialize CACLI
        self.cacli("deinitialize")
        if not self.emulator:
            self.close_pipe()
        self.initialized = False

    def read_pos_file(self) -> Tuple[npt.NDArray[np.float64],npt.NDArray[np.float64],list[bool]]:
        """ Read out the previously saved position file for consistency checking.
            This file contains the previous user position, zero offset and what
            axis was zeroed. All of which is returned as two arrays and a list.

            The file location is set by self.pos_file.

        Returns
        -------
        tuple(np.ndarray[3,float],np.ndarray[3,float],list[bool])
            The previous user position, zero position, and zero status.
        """
        """
        Example file:
            Register: User Position
            -793.384525
            -689.583077
            -7.500000
            Register: Zero Values
            756.501169
            760.263077
            8.100000
            Register: Zero status (XY/Z)
            1
            0
        """
        # If this file doesn't exist, we initialize a new one
        if not self.pos_file.exists():
            # Make the directory the file should be in
            self.pos_file.parents[0].mkdir(parents=True, exist_ok=True)
            # Make the file
            self.pos_file.touch()
            log.warn("File does not exist, creating and using all zero.")
            return (np.array([0,0,0],dtype=float), 
                    np.array([0,0,0],dtype=float), 
                    [False,False])
        else:
            try:
                # Open the file and read the lines
                with self.pos_file.open() as f:
                    lines = [f.readline() for _ in range(11)]
                # Extract relevant data (see example above)
                user_pos = [float(line.strip()) for line in lines[1:4]]
                zero_pos = [float(line.strip()) for line in lines[5:8]]
                zeroing = [bool(int(line.strip())) for line in lines[9:11]]
                return np.array(user_pos),np.array(zero_pos),np.array(zeroing)

            except ValueError:
                log.warn("Could not properly read position file, using all zero.")
                return (np.array([0,0,0],dtype=float), 
                        np.array([0,0,0],dtype=float), 
                        [False,False])

    def write_pos_file(self) -> None:
        """Write the current user position, zero position, and zero status. To
        the file specified in self.pos_file.
        """
        position = self.position
        zero = self.zero_position
        zeroing = self.zeroing
        #write values
        if not self.pos_file.exists():
            # Make the directory the file should be in
            self.pos_file.parents[0].mkdir(parents=True, exist_ok=True)
            # Make the file
            self.pos_file.touch()
        with self.pos_file.open('w') as f:
            f.write("Register: User Position\n")
            for pos in position:
                f.write(f"{pos:.8f}\n")
            f.write("Register: Zero Values\n")
            for zpos in zero:
                f.write(f"{zpos:.8f}\n")
            f.write("Register: Zero status (XY/Z)\n")
            for zset in zeroing:
                val = 1 if zset else 0
                f.write(f"{val}\n")

    def open_pipe(self) -> None:
        """Opens a server to the cacli executable, saving time on multiple
        commands. Otherwise, the usb connection is opened and closed between
        every call.
        """

        self.pipe = Popen([self.exe,f"SERV:{self.serial}"],creationflags=CREATE_NEW_CONSOLE)
        print("Opening pipe, please wait")
        sleep(4)
        if self.pipe.poll() is not None:
            raise RuntimeError("Pipe exited immediately, is this a duplicate?")
        else:
            return

    def close_pipe(self) -> None:
        """Closes the connection started by `open_pipe()`.
        """
        self.pipe.terminate()
        self.pipe.wait()

    def cacli(self, command:str, *args:list[str]) -> str:
        """Low level cacli access command, through which all other stage
        control/read are run through. Returns the message that cacli sends
        which can be further processed for reading the stage.
        This calls the cacli exe set by self.exe with the message structure
        "cacli.exe <command> <args[0]> <args[1]> ... <args[-1]>.
        This function also contains a bit of syntactic sugar for easier to
        understand base commands.

        Parameters
        ----------
        command : str
            The main command to send to cacli. Either the string to send
            directly or either one of ["initialize", "deinitialize", "set", "get", "stop"]
            which will be converted to the less transparant cacli version.
        
        *args : str
            A list of strings or string convertable objects to be appended after
            the main command in the cacli execution call. Each element will have
            a space added in between its neighbours.
            
        Returns
        -------
        str
            The full string response of the cacli executable.
        """
        if not self.initialized and not (command in ['FBEN','initialize']):
            raise RuntimeError("Stage not Initialized")
        commands = {"initialize"   : 'FBEN',
                    "deinitialize" : 'FBXT',
                    "set"          : 'FBCS',
                    "get"          : 'FBST',
                    "stop"         : 'FBES'}
        
        command = commands.get(command,command)
        string_args = [str(arg) for arg in args]

        if self.emulator:
            res = run([self.exe,command] + string_args,capture_output=True)
        else:
            res = run([self.exe, f"@SERV:{self.serial}", command] + string_args, 
                      capture_output=True)

        if command != "FBST":
            log.debug(f"Sending cacli message {' '.join([command] + string_args)}")
        msg = res.stdout.strip()
        msg = msg.strip().decode('UTF-8')
        if "UNAUTHORIZED COMMAND" in msg:
            log.warn(f"'{command}' is invalid in the current context. cacli returned '{msg}'")
        one_liner = msg.replace("\r\n", " ")
        if not("POSITION CONTROL" in msg):
            log.debug(f"Got cacli message '{one_liner}'")
        return msg

    def get_status(self) -> dict[str,int]:
        """Gets the status of the stage, which comprisses initialization status,
        motion status, set positions, error values from those positions.

        Returns
        -------
        dict[str,int]
            A dictionary of the status values, with each key the name, and each
            value the corresponding value.
        """
        for i in range(4):
            try:
                msg = self.cacli('get')
                lines = msg.splitlines()
                splits = [line.split(':') for line in lines[1:]]
                state = {split[0] : int(split[1]) for split in splits}
                return state
            except ValueError as e:
                print("Error occured while getting status, trying again:")
                print(e)
            except IndexError as e:
                print("Error occured while getting status, trying again:")
                print(e)

        raise RuntimeError("Could not get stage status.")
    
    @property
    def clicks(self) -> npt.NDArray[np.int]:
        """Get the clicks of each actuator position.
        This is offset by the clicks incase of controller zeroing.

        Returns
        -------
        np.array(3,int)
            The array of Stepper1,Stepper2,Stepper3 position in clicks.
        """
        state = self.get_status()
        clicks = np.array([state[name] for name in ['POS1','POS2','POS3']]).astype(int)
        clicks += self.offset_clicks
        return clicks

    @property
    def _raw_clicks(self) -> npt.NDArray[np.int]:
        """Get the raw clicks of each actuator position.
        Mostly used for debugging reasons.

        Returns
        -------
        np.array(3,int)
            The array of Stepper1,Stepper2,Stepper3 raw position in clicks.
        """
        state = self.get_status()
        raw_clicks = np.array([state[name] for name in ['POS1','POS2','POS3']]).astype(int)
        return raw_clicks        

    @property
    def position(self) -> npt.NDArray[np.float64]:
        """Get the current cartesian position of the stage in the user
        reference frame, converted to microns.

        Returns
        -------
        np.array(3,float)
            The position of the stage in cartesian coordinates (x,y,z) in microns.
        """
        z_clicks = self.clicks
        z_um = [self.clicks_to_microns(click) for click in z_clicks]
        stage_pos = stp_conv.cart_from_zs(z_um)

        pos = self.stage_to_user(stage_pos)

        return pos

    @property
    # Stage position with offset to match initialization, but no zeroing.
    def abs_position(self) -> npt.NDArray[np.float64]:
        """Get the current cartesian position of the stage in the user
        reference frame, ignoring zeroing, converted to microns.

        Returns
        -------
        np.array(3,float)
            The position of the stage in cartesian coordinates (x,y,z) in microns.
        """
        z_clicks = self.clicks
        z_um = [self.clicks_to_microns(click) for click in z_clicks]
        stage_pos = stp_conv.cart_from_zs(z_um)
        
        return stage_pos

    def set_z_clicks(self,z_clicks:npt.NDArray[np.int]) -> str:
        """Wrapper for setting the z_clicks with the appropriate offset
        to account for mismatches between controller clicks and known position.
        Essentially any command that moves the stages should absolutely use this
        command for proper position tracking. Otherwise, unexpected offsets may occur.

        Parameters
        ----------
        z_clicks : npt.NDArray[np.int]
            A set of clicks representing the absolute position of the stage
            desired. Even if the controller gets switched off in between uses.

        Returns
        -------
        str
            The return message from cacli, for ensuring the command was properly
            receieved. 
        """
        offset_corrected_clicks = z_clicks - self.offset_clicks
        msg = self.cacli('set', *offset_corrected_clicks)
        return msg

    def set_position(self,x:float=None,y:float=None,z:float=None,
                     monitor:bool=True, write_pos:bool=True,
                     monitor_kwargs:dict = {}, dry_run=False) -> None:
        """Set the stage position according to the given point in the user
        reference frame. Will check if the motion is safe, and if needed, also
        back out the stage in z before moving in xy and then returning to the
        desired z position.

        Parameters
        ----------
        x : float, optional
            The desired x position in microns in the user frame,
            by default, will use the current value of the user set position.
        y : float, optional
            The desired y position in microns in the user frame,
            by default, will use the current value of the user set position.
        z : float, optional
            The desired z position in microns in the user frame,
            by default, will use the current value of the user set position.
        monitor : bool, optional
            Wether to montor the motion, polling the stage position and printing 
            it while moving before returning from the function, by default False
        write_pos : bool, optional
            Wether to write the position of the stage to the position file after 
            the motion only applicable if monitor=True, by default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        dry_run : bool, optional
            If true, will run through the calculations of the stage motion
            without actually moving the stage or writing to the position 
            file. By default False.
        """
        prefix = ""
        if dry_run:
            prefix += "Dry Run - "
        # Replace None values with previously set user position
        # We avoid using the current position in cases where the xyz value
        # changes slightly from the desired position due to rounding.
        # This avoids confusion with repeated relative motion.
        new_pos = [x,y,z]
        new_pos = [pos if pos is not None else self.user_set_position[i] 
                   for i,pos in enumerate(new_pos)]
        new_pos = np.array(new_pos,dtype=float)
        # If we're not actually moving, skip the motion steps
        if np.all(new_pos == self.user_set_position):
            log.info(f"{prefix}New user position same as current, skipping move")
            return

        self.enforce_limits(new_pos, self.position)
        log.debug(f"{prefix}New user position: {new_pos}")

        # Convert from User Position to Stage Position
        new_stage_pos = self.user_to_stage(new_pos)
        log.debug(f"{prefix}New stage position: {new_stage_pos}")
        # Convert cartesian microns to z clicks
        z_um = stp_conv.zs_from_cart(new_stage_pos)
        log.debug(f"{prefix}New z microns: {z_um}")
        z_clicks = self.microns_to_clicks(z_um)
        log.debug(f"{prefix}New clicks: {z_clicks}")

        # If we're not actually moving, skip the motion steps
        if np.all(z_clicks == self.clicks):
            log.info(f"{prefix}New clicks same as current, skipping move")
            return

        ## FOR TESTING
        log.warn(f"{prefix}New position set to {new_pos}.")
        # Calculate the lowest possible z position along the path
        low_z = self.lowest_z(z_clicks)
        # We'll compare this to what the shortest allowed position should be
        max_z = self.lims[2][1]
        # If the biggest z value (shortest position) is larger then the allowable
        # value, by more than the stepper resolution we'll take extra care
        if low_z - max_z > self.res[2]:
            self.compensate_z_move(new_pos,max_z,low_z,monitor=monitor,write_pos=monitor,
                                   monitor_kwargs = monitor_kwargs, dry_run=dry_run)
            # All motion is now completed, so we can return.
            return
        
        # If instead, we're not changing the z position by a big amount,
        # we proceed as normal.
        if dry_run:
            log.debug(f"{prefix}Completed, returning without motion.")
            return

        # If we made it here without error
        # then we'll save the new user_set_position
        # This will be done in the recursive calls performed in the bad z, xy
        # motion case.
        self.user_set_position = np.copy(new_pos)
        log.debug(f"Set {self.user_set_position = }")
        # Set the position and make sure response is valid.
        msg = self.set_z_clicks(z_clicks)
        if msg != "STATUS : POSITION CONTROL SET":
            log.warn(f"Cacli returned unexpected result: {msg}")
        log.debug(f"Set new clicks to {z_clicks}, with position {new_pos}.")
        # Monitor then write if needed.
        # Writing position relies on stage not being in motion
        # So can only write if we bother to wait for the motion to be done
        # This can also be done manually later on.
        if monitor:
            self.monitor_move(write_pos,**monitor_kwargs)
        ## For Testing
        log.warn(f"After motion position is {self.position}.")

    # To calculate the lowest Z possible during a motion, we take the target
    # position and, for each actuator, set the number of clicks to the 
    # maximum (shortest) that will be encountered during the motion.
    # Then from this new clicks array, we compute the stage position and return
    # the z value. This should correspond to the shortest possible cavity attained
    # during the motion of the stage, assuming the worst condition.
    def lowest_z(self,new_clicks:npt.NDArray[np.int]) -> float:
        """Calculate the lowest possible z position between the position
        set by `new_clicks` and the current position in clicks.

        Parameters
        ----------
        new_clicks : npt.NDArray[np.int]
            The position we'd like to move to, in actuator clicks.

        Returns
        -------
        float
            The lowest (biggest) z position that could be reached during the 
            motion. In most cases, likely to be a worst case scenario.
        """
        # Get the shortest cavity clicks.
        old_clicks = self.clicks
        min_clicks = np.maximum(new_clicks,old_clicks)

        # Compute the position.
        z_um = self.clicks_to_microns(min_clicks)
        stage_pos = stp_conv.cart_from_zs(z_um)
        user_pos = self.stage_to_user(stage_pos)

        # Return the z value.
        return user_pos[2]
    
    def compensate_z_move(self,new_pos:npt.NDArray[np.float64],
                          max_z:float,low_z:float,
                          monitor:bool=False,write_pos:bool=True,
                          monitor_kwargs:dict = {}, dry_run=False) -> None:
        """Perform an xyz motion while compensating for any sag during
        the change in xy position. First moves up in z by the amount needed
        to avoid bad stuff, then moves in xy, and finally returns to the desired
        z position. 

        Parameters
        ----------
        new_pos : np.ndarray
            The position array xyz in the user reference frame, in microns, where
            we'd like to go.
        max_z : float
            The biggest value of z that is within bounds, and acceptable to move to.
        low_z : float
            The lowest z position that the motion in xy will cause. Should 
            be calculated by `lowest_z()`.
        monitor : bool, optional
            For each step, wether to montor the motion, polling the stage 
            position and printing it while moving before returning from the 
            function, by default False
        write_pos : bool, optional
            For each step, wether to write the position of the stage to the 
            position file after the motion only applicable if monitor=True, by 
            default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        """
        prefix=""
        if dry_run:
            prefix += "Dry Run - "
        # First compute by how much we should move, it should always be
        # at least one click
        delta = np.min([max_z - low_z,-self.res[2]])
        # Then, convert the distance to clicks and back, taking the floor
        # So that we always round to a larger number of clicks upwards.
        delta = np.floor(delta / self.um_conv) * self.um_conv
        # Since the relative limits might be smaller than this amount in z.
        # We temporarily override them
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # WARNING THIS ASSUMES THAT YOU NEVER ACTUALLY WORRY ABOUT MOVING
        # UP TOO MUCH, THIS IS DUE TO THE CURRENT EXPERIMENTAL DESIGN THAT IS
        # VERY UNLIKELY TO CHANGE!!!!!
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # Get relative limits
        rel_lims = self.rel_lims
        if rel_lims[2] < np.abs(delta):
            new_lims = rel_lims.copy() # Make a copy
            # Set the new limit to how much we're going to move
            new_lims[2] = max([np.abs(delta),np.abs(delta) + (new_pos[2]-self.position[2])])
            self.rel_lims = new_lims
        # Notify user that we're moving
        log.info(f"{prefix}Shortest z position {low_z} is > {max_z}, moving z by {delta} first.")
        # First move up by the calculate delta.
        self.move_rel(0,0,delta,monitor=monitor,write_pos=write_pos,
                     monitor_kwargs = monitor_kwargs, dry_run=dry_run)
        log.debug(f"{prefix}Compensated z position: {self.position = }, {self.user_set_position = }")
        # Then do the xy move, keeping the z position to what we just set it as
        log.info(f"{prefix}Performing compensated z move to xy = ({new_pos[0]},{new_pos[1]})")
        self.set_position(new_pos[0],new_pos[1],None,monitor=monitor,write_pos=write_pos,
                           monitor_kwargs = monitor_kwargs, dry_run=dry_run)
        log.debug(f"{prefix}Non decompensated position: {self.position}, {self.user_set_position = }")
        # Then, set the z position to what the original movement wanted.
        log.info(f"{prefix}Returning to desired z position {new_pos[2]}.")
        log.debug(f"{prefix}Final Position {self.position}, {self.user_set_position = }")
        self.set_position(None,None,new_pos[2],monitor=monitor,write_pos=write_pos,
                          monitor_kwargs = monitor_kwargs, dry_run=dry_run)
        # Then, reset the relative limits to what they were
        self.rel_lims = rel_lims

    @property
    def error(self) -> npt.NDArray[np.int]:
        """Get the current error on all the actuators.

        Returns
        -------
        npt.NDArray[np.int]
            The error, i.e. clicks from the set number of clicks. On each
            actuator. Should always be zero if the stage isn't moving.
        """
        state = self.get_status()
        return [state[name] for name in ['ERR1','ERR2','ERR3']]

    def enforce_limits(self,new_pos:npt.NDArray[np.float64],
                            cur_pos:npt.NDArray[np.float64]) -> None:
        """Checks the new position compared to the current position and does
        nothing if all the bounds are respected. Otherwise, raises an error.
        There are three types of bounds to follow. Relative bounds set by
        self.rel_lim, the stage can't move further than this amount along each
        axis in a single step.
        Absolute bounds set by self.lim, the stage can't move outside these limits
        along each axis.
        Hard bounds, set by the physical properties of the stage. No Actuator 
        position in clicks can go above 0 (always negative).

        Parameters
        ----------
        new_pos : npt.NDArray[np.float64]
            The position we would like to move to, in the user reference frame.
        cur_pos : npt.NDArray[np.float64]
            The current user set position.

        Raises
        ------
        JPEPositionError
            Raised if the new position violates any of the three possible bounds.
            Relative, Absolute, Hard. Error message contains any relevant
            information.
        """
        axis = ['x','y','z']
        change = np.abs(np.array(new_pos) - np.array(cur_pos))
        for i, delta in enumerate(change):
            if delta > (self.rel_lims[i] + self.res[i]/2):
                msg = (f"{axis[i]} rel position invalid. Change {delta} "
                                  f"larger than relative limit {self.rel_lims[i]}")
                log.error(msg)
                raise JPEPositionError(msg)

        for i, pos in enumerate(new_pos):
            if not (self.lims[i][0] - self.res[i]/2 <= pos <= self.lims[i][1] + self.res[i]/2):
                msg = (f"{axis[i]} abs position invalid. Position {pos} "
                                  f"outside range {self.lims[i]}")
                log.error(msg)
                raise JPEPositionError(msg)

        # Convert from User Position to Stage Position
        abs_pos = self.abs_position
        if not stp_conv.check_bounds(abs_pos[0],abs_pos[1],abs_pos[2]):
            msg = (f"Absolute position {abs_pos} outside hard limits.")
            log.error(msg)
            raise JPEPositionError(msg)

    def move_rel(self,x:float = 0.0, y:float = 0.0, z:float = 0.0,
                 monitor:bool = True, write_pos:bool = True,
                 monitor_kwargs:dict = {}, dry_run=False) -> None:
        """
        Move a relative distance from the current position.
        x,y,z set the distance to travel in the cartesian coordinates of the stage.
        This is done by updating the user_set_pos and then moving to this new 
        position, this way, imprecision in the stage will still result in +motion -motion
        giving a net zero position change.

        The position change will snap to the nearest reachable point. So a change
        of less than half the resolution will not move, while a change within 0.5-1 * resolution
        will result in a single step of motion.

        Parameters
        ----------
        x : float, optional
            Distance in microns to move along the x-axis, by default 0.0
        y : float, optional
            Distance in microns to move along the y-axis, by default 0.0
        z : float, optional
            Distance in microns to move along the z-axis, by default 0.0

        monitor : bool, optional
            Wether to montor the motion, polling the stage 
            position and printing it while moving before returning from the 
            function, by default False
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion only applicable if monitor=True, by 
            default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        """
        prefix = ""
        if dry_run:
            prefix += "Dry Run - "
        new_pos = list(self.user_set_position + np.array([x,y,z]))
        for i, rel in enumerate([x,y,z]):
            if np.abs(rel) <= self.res[i]/2:
                new_pos[i] = None
        log.info(f"{prefix}Moving {np.array([x,y,z])} um along each axis.")

        self.set_position(*new_pos, monitor, write_pos, monitor_kwargs=monitor_kwargs,dry_run=dry_run)

    def move_rel_xy(self,x:float = 0.0, y:float = 0.0,
                 monitor:bool = True, write_pos:bool = True,
                 monitor_kwargs:dict = {}, dry_run=False) -> None:
        """
        Move a relative distance from the current position.
        x,y,z set the distance to travel in the cartesian coordinates of the stage.
        This is done by updating the user_set_pos and then moving to this new 
        position, this way, imprecision in the stage will still result in +motion -motion
        giving a net zero position change.

        The position change will snap to the nearest reachable point. So a change
        of less than half the resolution will not move, while a change within 0.5-1 * resolution
        will result in a single step of motion.

        Parameters
        ----------
        x : float, optional
            Distance in microns to move along the x-axis, by default 0.0
        y : float, optional
            Distance in microns to move along the y-axis, by default 0.0

        monitor : bool, optional
            Wether to montor the motion, polling the stage 
            position and printing it while moving before returning from the 
            function, by default False
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion only applicable if monitor=True, by 
            default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        """
        prefix = ""
        if dry_run:
            prefix += "Dry Run - "

        new_pos = list(self.user_set_position + np.array([x,y,0.0]))
        for i, rel in enumerate([x,y]):
            if np.abs(rel) <= self.res[i]/2:
                new_pos[i] = None
        # Never moving in z
        new_pos[2] = None
        log.info(f"{prefix}Moving {np.array([x,y])} um along (x,y) axes.")

        self.set_position(*new_pos, monitor, write_pos, monitor_kwargs=monitor_kwargs,dry_run=dry_run)

    def move_up(self,z:float = 0.0,
                monitor:bool = True, write_pos:bool = True,
                monitor_kwargs:dict = {}, dry_run=False) -> None:
        """
        Move a relative distance up from the current position.
        z sets the distance to travel upwards.
        This is done by updating the user_set_pos and then moving to this new 
        position, this way, imprecision in the stage will still result in +motion -motion
        giving a net zero position change.

        The position change will snap to the nearest reachable point. So a change
        of less than half the resolution will not move, while a change within 0.5-1 * resolution
        will result in a single step of motion.

        Parameters
        ----------
        z : float, optional
            Distance in microns to move upwards, must be positive, by default 0.0

        monitor : bool, optional
            Wether to montor the motion, polling the stage 
            position and printing it while moving before returning from the 
            function, by default False
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion only applicable if monitor=True, by 
            default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        """
        prefix = ""
        if dry_run:
            prefix += "Dry Run - "
        if z < 0.0:
            raise JPEPositionError("z value in move up must be positive.")
        ##################################################################
        # NEGATIVE VALUES IS GOING UP. ABSOLUTELY CORRECT DO NOT CHANGE! #
        new_pos = list(self.user_set_position - np.array([0.0,0.0,z]))   #
        ##################################################################
        if z <= self.res[2]/2:
            new_pos[2] = None
        # Never moving in xy
        new_pos[0] = None
        new_pos[1] = None 
        log.info(f"{prefix}Moving {z} um upwards.")

        self.set_position(*new_pos, monitor, write_pos, monitor_kwargs=monitor_kwargs,dry_run=dry_run)

    def move_down(self,z:float = 0.0,
                  monitor:bool = True, write_pos:bool = True,
                  monitor_kwargs:dict = {}, dry_run=False) -> None:
        """
        Move a relative distance down from the current position.
        z sets the distance to travel downwards.
        This is done by updating the user_set_pos and then moving to this new 
        position, this way, imprecision in the stage will still result in +motion -motion
        giving a net zero position change.

        The position change will snap to the nearest reachable point. So a change
        of less than half the resolution will not move, while a change within 0.5-1 * resolution
        will result in a single step of motion.

        Parameters
        ----------
        z : float, optional
            Distance in microns to move downwards, must be positive, by default 0.0

        monitor : bool, optional
            Wether to montor the motion, polling the stage 
            position and printing it while moving before returning from the 
            function, by default False
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion only applicable if monitor=True, by 
            default True
        monitor_kwargs : dict, optional
            Extra keywords to pass to the monitor function.
        """
        prefix = ""
        if dry_run:
            prefix += "Dry Run - "
        if z < 0.0:
            raise JPEPositionError(f"{prefix}z value in move down must be positive.")
        ####################################################################
        # POSITIVE VALUES IS GOING DOWN. ABSOLUTELY CORRECT DO NOT CHANGE! #
        new_pos = list(self.user_set_position + np.array([0.0,0.0,z]))     #
        ####################################################################
        if z <= self.res[2]/2:
            new_pos[2] = None
        # Never moving in xy
        new_pos[0] = None
        new_pos[1] = None 
        log.info(f"{prefix}Moving {z} um downwards.")

        self.set_position(*new_pos, monitor, write_pos, monitor_kwargs=monitor_kwargs, dry_run=dry_run)

    def monitor_move(self, write_pos:bool = True, 
                     display_callback:Callable = None, 
                     abort_callback:Callable = lambda *args: False) -> int:
        """Monitor the stage for the current motion. Until either a reason
        to abort is encountered, or until the stage no longer reports that it
        is in motion. Built into this function, the motion will be aborted if 
        any of the error values remain constant over three polling loops, 
        indicating that one of the axis is stuck.

        The user can also supply their own function for checking wether to abort.
        This should return true when the user wants to abort.

        Parameters
        ----------
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion is complete, by default True
        display_callback : Callable, optional
            The function to call every loop to display progress.
            This function is passed the full status dictionary of the stage.
            By default lambda *args:None
        abort_callback : Callable, optional
            The function which checks wether or not to abort. 
            Return false to continue the motion or true to emergency stop the 
            stage. This function is passed the full status dictionary of the stage. 
            By default lambda *args:False

        Returns
        -------
        int
            -1 if the motion was aborted, 0 otherwise.
        """
        prev_error = np.array([0,0,0])
        stuck_count = 0
        aborted = False
        if display_callback is None:
            display_callback = self.disp_status
        while True:
            # Get the status of the stage
            status = self.get_status()

            # Check if we should abort motion
            if abort_callback(status):
                self.cacli('stop')
                aborted = True
                break
            
            # Display the current status
            display_callback(status)

            # Get out the current error values (clicks from setpoint).
            error = np.array([status[name] for name in ['ERR1','ERR2','ERR3']])
            # If we're not moving, break the loop, we're done
            if status['BUSY'] == 0:
                # However, if any of the errors is not zero, and we're not moving
                # something weird has happened.
                if np.any(error > 0):
                    aborted = True
                break
            
            # If we're moving, but the error isn't changing after a few loops
            # Something is up, and we should abort.
            if np.any(np.logical_and(error == prev_error,error!=0)):
                stuck_count += 1
            else:
                stuck_count = 0

            if stuck_count > self.stuck_iterations:
                self.cacli('stop')
                aborted = True
                break
            prev_error = error

        if aborted:
            log.warn("Something went wrong while moving, motion aborted.")
            self.error_flag = True
            return -1
        else:
            if write_pos:
                self.write_pos_file()

        return 0

    def async_monitor_move(self, write_pos:bool = True, 
                           display_callback:Callable = lambda *args: None, 
                           abort_callback:Callable = lambda *args: False) -> Thread:
        """From a new thread, monitors the stage for the current motion. 
        Until either a reason to abort is encountered, or until the stage 
        no longer reports that it is in motion. 
        Built into this function, the motion will be aborted if 
        any of the error values remain constant over three polling loops, 
        indicating that one of the axis is stuck.

        The user can also supply their own function for checking wether to abort.
        This should return true when the user wants to abort.

        Parameters
        ----------
        write_pos : bool, optional
            Wether to write the position of the stage to the 
            position file after the motion is complete, by default True
        display_callback : Callable, optional
            The function to call every loop to display progress.
            This function is passed the full status dictionary of the stage.
            By default lambda *args:None
        abort_callback : Callable, optional
            The function which checks wether or not to abort. 
            Return false to continue the motion or true to emergency stop the 
            stage. This function is passed the full status dictionary of the stage. 
            By default lambda *args:False

        Returns
        -------
        Thread
            The thread in which the function is running.
        """
        t = Thread(target=self.monitor_move, 
                   args = (display_callback,abort_callback))
        t.start()
        return t

    def stop(self) -> None:
        """Emegency Stops the stage.
        """
        self.cacli("stop")

    def set_cryotemp(self) -> None:
        """Deinitialize and reinitialize the stage, changing the gain and
        temperature configure to be set for cryogenic operation.
        The stage should not be run at an intermediary condition, 
        either fully cold or fully ambient.
        """
        self.deinitialize()
        self.gain = 300
        self.temp = 10
        self.stuck_iterations = 100
        self.initialize()
        log.info("Set Gain = 180, Temp = 10 for cryo operation.")

    def set_roomtemp(self) -> None:
        """Deinitialize and reinitialize the stage, changing the gain and
        temperature configure to be set for room temperature operation.
        The stage should not be run at an intermediary condition, 
        either fully cold or fully ambient.
        """
        self.deinitialize()
        self.gain = 60
        self.temp = 293
        self.stuck_iterations = 6
        self.initialize()
        log.info("Set Gain = 60, Temp = 293 for RT operation.")

    @property
    def rel_lims(self) -> npt.NDArray[np.float64]:
        """Get the relative motion limit for each axis.

        Returns
        -------
        list[float]
            The relative motion limit along each (x,y,z) axis.
        """
        return self._rel_lims
        
    @rel_lims.setter
    def rel_lims(self,rel_lims:npt.NDArray[np.float64]) -> None:
        """Set the relative motion limit for each axis.
        Must be greater than zero.

        Parameters
        ----------
        rel_lims : list[float]
            How much the stage should be limited to move by in one step, for
            each axis.

        Raises
        ------
        JPELimitError
            Error returned if any of the relative limits is less than zero.
            Sign doesn't matter for these, so keep positive.
        """
        axis = ['x','y','z']
        for i, lim in enumerate(rel_lims):
            if lim < 0:
                msg = (f"{axis[i]} invalid relative limit {lim}. Must be positive.")
                log.error(msg)
                raise JPELimitError(msg)
        self._rel_lims = rel_lims

    @property
    def lims(self) -> npt.NDArray[np.float64]:
        """Get the soft bounds on each axis' motion.

        Returns
        -------
        np.ndarray[(3,2),float]
            For each axis, the lower and upper z limit.
        """
        return self._lims
    @lims.setter
    def lims(self, limits:npt.NDArray[np.float64]) -> None:
        """Set the soft bounds on each axis' motion.

        Parameters
        ----------
        limits : np.ndarray[(3,2),float]
            A 2D array with a pair [min,max] of bounds for each axis.

        Raises
        ------
        ValueError
            Raises an error if any of the lower bounds are higher than
            the upper bounds.
        """
        axis = ['x','y','z']
        for i, lims in enumerate(limits):
            if lims[0] > lims[1]:
                msg = f"{axis[i]} invalid limits {lims}. Lower limit must be less than upper."
                log.error(msg)
                raise ValueError(msg)
        self._lims = limits

    def toggle_zero_xy(self) -> None:
        """Toggles between zeroing the xy stage position or not.
        If turning on, we record the current xy position and start
        using that as a position offset, and set the user_set_position to 0
        in both x and y.
        If turning off, we simply toggle the value and reset the user_set_position
        to the current values in x and y.
        """
        if self.zeroing[0]:
            log.debug("Disable XY Zeroing")
            self.zeroing[0] = False
            self.user_set_position[0] = self.position[0]
            self.user_set_position[1] = self.position[1]
            log.debug(f"User Position Now {self.user_set_position}")
        else:
            log.debug("Enabling XY Zeroing")
            self.zeroing[0] = True
            self.zero_position[0] += self.position[0]
            self.zero_position[1] += self.position[1]
            self.user_set_position[0] = 0.0
            self.user_set_position[1] = 0.0
            log.debug(f"User Position Now {self.user_set_position}")
        # Write status to file.
        self.write_pos_file()

    def toggle_zero_z(self) -> None:
        """Toggles between zeroin the z stage position or not.
        If turning on, we record the current z position and start
        using that as a position offset, and set the user_set_position to 0
        in z.
        If turning off, we simply toggle the value and reset the user_set_position
        to the current values in z.
        """
        if self.zeroing[1]:
            log.debug("Disabling Z Zeroing")
            self.zeroing[1] = False
            self.user_set_position[2] = self.position[2]
            log.debug(f"User Position Now {self.user_set_position}")
        else:            
            log.debug("Enabling Z Zeroing")
            self.zeroing[1] = True
            self.zero_position[2] += self.position[2]
            self.user_set_position[2] = 0.0
            log.debug(f"User Position Now {self.user_set_position}")
        # Write status to file.
        self.write_pos_file()

    def disp_status(self,state:dict) -> None:
        """A handy function for displaying the stage status, useful
           for monitoring the motion of the stage.

        Parameters
        ----------
        state : dict
            The dictonary returned by `self.get_status()` that contains the
            current stage position in clicks, as well as the error.
        """
        raw_clicks = np.array([state[name] for name in ['POS1','POS2','POS3']]).astype(int)
        errors = np.array([state[name] for name in ['ERR1','ERR2','ERR3']]).astype(int)
        z_um = self.clicks_to_microns(raw_clicks)
        e_um = self.clicks_to_microns(errors)
        stage_pos = stp_conv.cart_from_zs(z_um)
        stage_errors = stp_conv.cart_from_zs(e_um)
        user_pos = self.stage_to_user(stage_pos)
        print("Status Update:")
        print(f"\tUser:\t({', '.join([f'{val:.2f}' for val in user_pos])})")
        print(f"\tStage:\t({', '.join([f'{val:.2f}' for val in stage_pos])})")
        print(f"\tRaw Clicks:\t({', '.join([f'{val}' for val in raw_clicks])})")
        print(f"\tClicks:\t({', '.join([f'{val}' for val in raw_clicks + self.offset_clicks])})")
        print(f"\tError:\t({', '.join([f'{val}' for val in errors])})")
        print(f"\tE-(um):\t({', '.join([f'{val:.2f}' for val in stage_errors])})")

    def write_status(self,state:dict,filename:str,disp=True) -> None:
        """A handy function that writes the user position to a file
           from the stage status, useful for monitoring the motion of the stage.

        Parameters
        ----------
        state : dict
            The dictonary returned by `self.get_status()` that contains the
            current stage position in clicks, as well as the error.
        filename : str
            The directory of the file to which to write the position information.
        disp : bool, optional
            If True, will also call disp_status to display the stage
            position information for the current state. By default True.
        """
        clicks = np.array([state[name] for name in ['POS1','POS2','POS3']]).astype(int)
        z_um = self.clicks_to_microns(clicks)
        stage_pos = stp_conv.cart_from_zs(z_um)
        user_pos = self.stage_to_user(stage_pos)

        with open (filename,'a') as f:
            f.write(f"{time()}, {','.join([f'{pos:.2f}' for pos in user_pos])}\n")
        if disp:
            self.disp_status(state)

    def track_motion(self,x:float=None,y:float=None,z:float=None,
                     filename:str = "move.csv") -> None:
        with open(filename,'w') as f:
            f.write(f"{time()}, {','.join([f'{pos:.2f}' for pos in self.position])}\n")
        monitor_func = lambda state: self.write_status(state,filename=filename)
        self.set_position(x,y,z,monitor=True,
                          monitor_kwargs={'display_callback' : monitor_func})
        with open(filename,'a') as f:
            f.write(f"{time()}, {','.join([f'{pos:.2f}' for pos in self.position])}\n")
