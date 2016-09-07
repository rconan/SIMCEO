function SCEO(block)

setup(block);

function setup(block)

msg_box   = get(gcbh,'UserData');
fprintf('__ %s: SETUP __\n',msg_box.class_id)
% Register number of ports
%block.NumInputPorts  = 0;

% Setup port properties to be inherited or dynamic
%block.SetPreCompInpPortInfoToDynamic;
%block.SetPreCompOutPortInfoToDynamic;

IO_setup(msg_box, block)

% Register sample times
%  [0 offset]            : Continuous sample time
%  [positive_num offset] : Discrete sample time
%
%  [-1, 0]               : Inherited sample time
%  [-2, 0]               : Variable sample time
%block.SampleTimes = [1 0];

% Specify the block simStateCompliance. The allowed values are:
%    'UnknownSimState', < The default setting; warn and assume DefaultSimState
%    'DefaultSimState', < Same sim state as a built-in block
%    'HasNoSimState',   < No sim state
%    'CustomSimState',  < Has GetSimState and SetSimState methods
%    'DisallowSimState' < Error out when saving or restoring the model sim state
block.SimStateCompliance = 'DefaultSimState';

%% -----------------------------------------------------------------
%% The MATLAB S-function uses an internal registry for all
%% block methods. You should register all relevant methods
%% (optional and required) as illustrated below. You may choose
%% any suitable name for the methods and implement these methods
%% as local functions within the same file. See comments
%% provided for each function for more information.
%% -----------------------------------------------------------------

block.RegBlockMethod('Start', @Start);
block.RegBlockMethod('Outputs', @Outputs);     % Required
block.RegBlockMethod('Update', @Update);
block.RegBlockMethod('Terminate', @Terminate); % Required
block.RegBlockMethod('PostPropagationSetup', @PostPropagationSetup);
block.RegBlockMethod('InitializeConditions', @InitializeConditions);
%end setup

function PostPropagationSetup(block)
msg_box   = get(gcbh,'UserData');
fprintf('__ %s: PostPropagationSetup __\n',msg_box.class_id)

function InitializeConditions(block)
msg_box   = get(gcbh,'UserData');
fprintf('__ %s: InitializeConditions __\n',msg_box.class_id)
deal(msg_box,block,'init')

function Start(block)

msg_box   = get(gcbh,'UserData');
fprintf('__ %s: START  __\n',msg_box.class_id)
deal(msg_box,block,'start')
%set(gcbh,'UserData',msg_box)
tic
%end Start

function Outputs(block)

msg_box   = get(gcbh,'UserData');
%fprintf('__ %s: OUTPUTS __\n',msg_box.class_id)

deal(msg_box,block,'IO')

%end Outputs

function Update(block)

%end Update

function Terminate(block)

toc
msg_box = get(gcbh,'UserData');
deal(msg_box,block,'terminate')
set(gcbh,'UserData',[])
%end Terminate
