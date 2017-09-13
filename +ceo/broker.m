classdef (Sealed=true) broker < handle
    % broker An interface to a CEO server
    %  The broker class launches an AWS instance and sets up the connection
    %  to the CEO server
    
    properties
        awspath % full path to the AWS CLI
        instance_id % The AWS instance ID number
        public_ip % The AWS instance public IP
        zmqReset % ZMQ connection reset flag
        elapsedTime
    end
    
    properties (Access=private)
        etc
        instance_end_state
        ctx
        socket
    end
    
    methods

        function self = broker(varargin)

            self.ctx    = zmq.core.ctx_new();
            self.socket = zmq.core.socket(self.ctx, 'ZMQ_REQ');
            self.zmqReset = true;
            
            self.elapsedTime = 0;

            currentpath = mfilename('fullpath');
            k = strfind(currentpath,filesep);
            self.etc = fullfile(currentpath(1:k(end)),'..','etc');
            cfg = loadjson(fullfile(self.etc,'simceo.json'));
            self.awspath         = cfg.awsclipath;
            self.instance_id     = cfg.aws_instance_id;
            if isempty(self.instance_id)
                run_instance(self)
                self.instance_end_state = 'terminate';
            else
                start_instance(self)
                self.instance_end_state = 'stop';
            end
        end

        function delete(self)
            fprintf('@(broker)> Deleting %s\n',class(self))
            terminate_instance(self)    
            zmq.core.close(self.socket);
            zmq.core.ctx_shutdown(self.ctx);
            zmq.core.ctx_term(self.ctx);
        end

        function run_instance(self)
            cmd = sprintf(['%s ec2 run-instances --profile gmto.control ',...
                           '--cli-input-json file://%s'],...
                          self.awspath, fullfile(self.etc,'ec2runinst.json'));
            [status,instance_json] = system(cmd);
            if status~=0
                error('Launching AWS AMI failed:\n%s',instance_json)
            end
            instance = loadjson(instance_json);
            self.instance_id = instance.Instances{1}(1).InstanceId;
            fprintf('>>>> WAITING FOR AWS INSTANCE %s TO START ... \n',self.instance_id)
            tic
            [status,~] = system(sprintf(['%s ec2 wait instance-running --instance-ids %s',...
                                ' --profile gmto.control'],...
                                        self.awspath,self.instance_id));
            toc
            if status~=0
                error('Starting AWS machine %s failed!',self.instance_id')
            end
            fprintf('>>>> WAITING FOR AWS INSTANCE %s TO INITIALIZE ... \n',self.instance_id)
            fprintf('(This usually takes a few minutes!)\n')
            tic
            cmd = sprintf(['%s ec2 wait instance-status-ok --instance-ids %s ',...
                           '--profile gmto.control'],...
                          self.awspath,self.instance_id);
            [status,~] = system(cmd);
            toc
            if status~=0
                error('Starting AWS machine %s failed!',self.instance_id')
            end
            [~,username] = system('whoami');
            [~,hostname] = system('hostname');
            cmd = sprintf('%s ec2 create-tags --resources %s --tags Key=Name,Value=%s',...
                          self.awspath,self.instance_id,...
                          ['SIMCEO(',strtrim(username),...
                           '@',strtrim(hostname),')']);
            system(cmd);
            cmd = sprintf(['%s cloudwatch put-metric-alarm ',...
                           '--profile gmto.control ',...
                           '--dimensions Name=InstanceId,Value=%s ',...
                           '--cli-input-json file://%s'],...
                          self.awspath,...
                          self.instance_id,...
                          fullfile(self.etc,'cloudwatch.json'));
            [status,~] = system(cmd);
            if status~=0
                error('Setting alarm for AWS machine %s failed!',self.instance_id')
            end
            cmd = sprintf(['%s ec2 describe-instances --instance-ids %s',...
                           ' --output text',...  
                           ' --query Reservations[*].Instances[*].PublicIpAddress',...
                           ' --profile gmto.control'],...
                          self.awspath,self.instance_id);
            [status,public_ip_] = system(cmd);
            if status~=0
                error('Getting AWS machine public IP failed!')
            end
            self.public_ip = strtrim(public_ip_);
            fprintf('\n ==>> machine is up and running @%s\n',self.public_ip)    
        end

        function start_instance(self)
            cmd = sprintf(['%s ec2 start-instances --instance-ids %s',...
                           ' --profile gmto.control'],...
                          self.awspath,self.instance_id);
            fprintf('%s\n',cmd)
            fprintf('@(broker)> Starting AWS machine %s...\n',self.instance_id)
            [status,cmdout] = system(cmd);
            if status~=0
                error('Starting AWS machine %s failed:\n%s',self.instance_id,cmdout)
            end
            fprintf('>>>> WAITING FOR AWS INSTANCE %s TO START ... \n',self.instance_id)
            tic
            [status,~] = system(sprintf(['%s ec2 wait instance-running --instance-ids %s',...
                                ' --profile gmto.control'],...
                                        self.awspath,self.instance_id));
            toc
            if status~=0
                error('Starting AWS machine %s failed!',self.instance_id')
            end
            cmd = sprintf(['%s ec2 describe-instances --instance-ids %s',...
                           ' --output text',...  
                           ' --query Reservations[*].Instances[*].PublicIpAddress',...
                           ' --profile gmto.control'],...
                          self.awspath,self.instance_id);
            [status,public_ip_] = system(cmd);
            if status~=0
                error('Getting AWS machine public IP failed!')
            end
            self.public_ip = strtrim(public_ip_);
            fprintf('\n ==>> machine is up and running @%s\n',self.public_ip)
        end

        function terminate_instance(self)
            if strcmp(self.instance_end_state,'terminate')
                fprintf('@(broker)> Terminating instance %s!\n',self.instance_id)
                [status,~] = system(sprintf(['%s ec2 %s-instances',...
                                    ' --instance-ids %s --profile gmto.control'],...
                                            self.awspath, self.instance_end_state,...
                                            self.instance_id));
                if status~=0
                    error('Terminating AWS instance %s failed!',self.instance_id')
                end
            end
        end
    end
    
    methods(Static)

        function self = getBroker(varargin)
        % getBroker Get a pointer to the broker object
        %
        % agent = ceo.broker.getBroker() % Launch an AWS instance and returns
        % a pointer to the broker object
        % agent = ceo.broker.getBroker('awspath','path_to_aws_cli') % Launch
        % an AWS instance using the given AWS CLI path and returns a pointer to
        % the broker object
        % agent =
        % ceo.broker.getBroker('instance_id','the_id_of_AWS_instance_to_start') 
        % Launch the AWS instance 'instance_id' and returns a pointer to the broker object
            
            persistent this
            if isempty(this)
                fprintf('~~~~~~~~~~~~~~~~~~~')
                fprintf('\n SIMCEO CLIENT!\n')
                fprintf('~~~~~~~~~~~~~~~~~~~\n')
                this = ceo.broker(varargin{:});
            end
            self = this;
        end

        function jmsg = sendrecv(send_msg)
            tid = tic;
            self = ceo.broker.getBroker();
            jsend_msg = saveubjson('',send_msg);
            zmq.core.send( self.socket, uint8(jsend_msg) );
            rcev_msg = -1;
            count = 0;
            while all(rcev_msg<0) && (count<15)
                rcev_msg = zmq.core.recv( self.socket , 2^24);
                if count>0
                    fprintf('@(broker)> sendrecv: Server busy (call #%d)!\n',15-count)
                end
                count = count + 1;
            end
            if count==15
                set_param(gcs,'SimulationCommand','stop')
            end
            jmsg = loadubjson(char(rcev_msg),'SimplifyCell',1);
            if ~isstruct(jmsg) && strcmp(char(jmsg),'The server has failed!')
                disp('Server issue!')
                set_param(gcs,'SimulationCommand','stop')
            end    
            self.elapsedTime = self.elapsedTime + toc(tid);
        end

        function resetZMQ()
            self = ceo.broker.getBroker();
            if self.zmqReset
                [~,aws_instance_state] = system(...
                    sprintf(['%s ec2 describe-instances --instance-ids %s',...
                             ' --output text',...
                             ' --query Reservations[*].Instances[*].State.Name ',...
                             '--profile gmto.control'],...
                    self.awspath, self.instance_id));
                if any(strcmp(strtrim(aws_instance_state),{'shutting-down','terminated'}))
                    run_instance(self)
                end
                zmq.core.close(self.socket);
                self.socket = zmq.core.socket(self.ctx, 'ZMQ_REQ');
                status = zmq.core.setsockopt(self.socket,'ZMQ_RCVTIMEO',60e3);
                if status<0
                    error('broker:zmqRcvTimeOut','Setting ZMQ_RCVTIMEO failed!')
                end
                status = zmq.core.setsockopt(self.socket,'ZMQ_SNDTIMEO',60e3);
                if status<0
                    error('broker:zmqSndTimeOut','Setting ZMQ_SNDTIMEO failed!')
                end
                address     = sprintf('tcp://%s:3650',self.public_ip);
                zmq.core.connect(self.socket, address);
                fprintf('@(broker)> %s connected at %s\n',class(self),address)
            end
            self.zmqReset = false;
        end
        function setZmqResetFlag(val)
            self = ceo.broker.getBroker();
            self.zmqReset = val;
        end

        function timeSpent()
            self = ceo.broker.getBroker();
            fprintf('@(broker)> Time spent communicating with the server: %.3fs\n',...
                                self.elapsedTime)
            self.elapsedTime = 0;
        end
        
    end

end
