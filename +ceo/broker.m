classdef (Sealed=true) broker < handle
    % broker An interface to a CEO server
    %  The broker class launches an AWS instance and sets up the connection
    %  to the CEO server
    
    properties
      ami_id % The AWS AMI ID number
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
        urlbase
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
            cfg = jsondecode(fileread(fullfile(self.etc,'simceo.json')));
            self.urlbase         = 'http://gmto.modeling.s3-website-us-west-2.amazonaws.com';
            self.ami_id          = cfg.aws_ami_id;
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
            zmq.core.close(self.socket);
            zmq.core.ctx_shutdown(self.ctx);
            zmq.core.ctx_term(self.ctx);
            url = sprintf('%s/simceo_aws_server.html?action=%s&instance_ID=%s',...
                          self.urlbase,self.instance_end_state,self.instance_id);
            fprintf('%s\n',url)
            [status,h] = web(url,'-browser');
            if status~=0
              error('Shutting down AWS machine %s failed:\n',self.instance_id)
            end
        end

        function run_instance(self)
          url = sprintf("%s/simceo_aws_server.html?action=create",self.urlbase);
          fprintf('%s\n',url)
          [status,h] = web(url,'-browser');
          if status~=0
            error('Creating machine failed:\n')
          end
          pause(20)
          url = sprintf('%s/%s.json',self.urlbase,self.ami_id);
          fprintf('%s\n',url)
          instance=jsondecode(char(webread(url))');
          self.instance_id = instance.ID;
          file = fullfile(self.etc,'simceo.json');
          cfg = jsondecode(fileread(file));
          cfg.aws_instance_id = instance.ID;
          savejson('',cfg,file)
          url = sprintf('%s/%s.json',self.urlbase,self.instance_id);
          fprintf('%s\n',url)
          instance=jsondecode(char(webread(url))');
          fprintf('STATE: %s\n',instance.STATE)
          n=1;
          while (~strcmp(instance.STATE,'running')) && (n<=3)
            fprintf('Probing instance state (20s wait time) ...\n')
            pause(20)
            instance=jsondecode(char(webread(url))');
            n = n + 1;
          end
          if (~strcmp(instance.STATE,'running')) && (n>3)
            error('Failed to start server!')
          end
          self.public_ip = instance.IP;
          fprintf('\n ==>> machine is up and running @%s\n',self.public_ip)
          %pause(2)
          %close(h)
        end

        function start_instance(self)
            fprintf('@(broker)> Starting AWS machine %s...\n',self.instance_id)

            url = sprintf('%s/simceo_aws_server.html?action=start&instance_ID=%s',self.urlbase,self.instance_id);
            fprintf('%s\n',url)
            [status,h] = web(url,'-browser');
            if status~=0
              error('Starting AWS machine %s failed:\n',self.instance_id)
            end
            pause(3)
            url = sprintf('%s/%s.json',self.urlbase,self.instance_id);
            fprintf('%s\n',url)
            instance=jsondecode(char(webread(url))');
            fprintf('STATE: %s\n',instance.STATE)
            n=1;
            while (~strcmp(instance.STATE,'running')) && (n<=3)
              fprintf('Probing instance state (20s wait time) ...\n')
              pause(20)
              instance=jsondecode(char(webread(url))');
              n = n + 1;
            end
            if (~strcmp(instance.STATE,'running')) && (n>3)
              error('Failed to start server!')
            end
            self.public_ip = instance.IP;
            fprintf('\n ==>> machine is up and running @%s\n',self.public_ip)
            %pause(2)
            %close(h)
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
            if isempty(this) || ~isvalid(this)
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
g
        function resetZMQ()
            self = ceo.broker.getBroker();
            if self.zmqReset
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
