classdef messages < handle

    properties
        n_in
        dims_in
        n_out
        dims_out
        start
        update
        outputs
        terminate
        init
        sampleTime
    end

    properties (Dependent)
       class_id
    end

    properties (Access=private)
        p_class_id
    end

    methods
        
            function self = messages(class_id)

                    self.p_class_id = class_id;
                    proto_msg = struct('class_id',self.p_class_id,...
                                       'method_id','',...
                                       'tag','',...
                                       'args',struct('args',[])); 
                    % Start
                    self.start     = proto_msg;
                    self.start.method_id = 'Start';
                    % InitializeConditions
                    self.init      = proto_msg;
                    self.init .method_id = 'InitializeConditions'; 
                    % Outputs
                    self.update    = proto_msg;
                    self.update.method_id  = 'Update';
                    self.outputs   = proto_msg;
                    self.outputs.method_id = 'Outputs';
                    % Terminate
                    self.terminate = proto_msg;
                    self.terminate.method_id = 'Terminate';
                end
                function val = get.class_id(self)
                    val = self.p_class_id;
                end
                function set.class_id(self,val)
                    self.p_class_id = val;
                    self.start.class_id     = val;
                    self.init.class_id      = val;
                    self.update.class_id    = val;
                    self.outputs.class_id   = val;
                    self.terminate.class_id = val;
                end
                function IO_setup(self,block)
                    block.NumInputPorts  = self.n_in;
                    for k_in=1:self.n_in
                        block.InputPort(k_in).Dimensions  = self.dims_in{k_in};
                        block.InputPort(k_in).DatatypeID  = 0;  % double
                        block.InputPort(k_in).Complexity  = 'Real';
                        block.InputPort(k_in).DirectFeedthrough = true;
                    end
                    block.NumOutputPorts = self.n_out;
                    for k_out=1:self.n_out
                        block.OutputPort(k_out).Dimensions   = self.dims_out{k_out};
                        block.OutputPort(k_out).DatatypeID   = 0; % double
                        block.OutputPort(k_out).Complexity   = 'Real';
                        block.OutputPort(k_out).SamplingMode = 'sample';
                    end
                    block.SampleTimes = self.sampleTime;
                end
                function deal(self,block,tag)
                    switch tag
                      case 'start'
                        deal_start(self);
                      case 'init'
                        deal_init(self);
                      case 'IO'
                        deal_inputs(self, block);
                        deal_outputs(self, block);
                      case 'terminate'
                        deal_terminate(self);
                      otherwise
                        fprintf(['@(messages)> deal: Unknown tag;',...
                                 ' valid tags are: start, init, IO and terminate!'])
                    end
                end
            
    end
    
    methods (Access=private)
        
                function deal_start(self)
                    ceo.broker.resetZMQ()
                    jmsg = ceo.broker.sendrecv(self.start);
                    tag = char(jmsg);
                    self.class_id = tag;
                    fprintf('@(%s)> Object created!\n',tag)
                end

                function deal_init(self)
                    jmsg = ceo.broker.sendrecv(self.init);
                    fprintf('@(messages)> Object calibrated!\n')
                end
                
                function deal_terminate(self)
                    jmsg = ceo.broker.sendrecv(self.terminate);
                    fprintf('@(%s)> %s\n',self.class_id,jmsg)
                    ceo.broker.setZmqResetFlag(true)
                end
                function deal_inputs(self, block)
                    if self.n_in>0
                        fields = fieldnames(self.update.args.inputs_args);
                        for k_in=1:self.n_in
                            self.update.args.inputs_args.(fields{k_in}) = ...
                                                  reshape(block.InputPort(k_in).Data,1,[]);
                        end
                    end
                    ceo.broker.sendrecv(self.update);
                end
                function deal_outputs(self, block)
                    if self.n_out>0
                        outputs_msg = ceo.broker.sendrecv(self.outputs);
                        fields = fieldnames(outputs_msg);
                        for k_out=1:self.n_out
                            data = outputs_msg.(fields{k_out});
                            if isempty(data)
                                data = NaN(size(block.OutputPort(k_out).Data));
                            end
                            block.OutputPort(k_out).Data = data;
                        end
                    end
                end

    end
 end
