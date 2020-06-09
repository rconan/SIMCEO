function mat2py(ssdatapath,varargin)

if nargin>1
    infile = fullfile(ssdatapath,varargin{1});
    [~,name,ext] = fileparts(infile);
    outfile = fullfile(ssdatapath,[name,'.py',ext]);
else
    infile = fullfile(ssdatapath,'modal_state_space_model.mat');
    outfile = fullfile(ssdatapath,'modal_state_space_model.py.mat');
end

if exist(outfile,'file')>0
    fprintf('%s already exist!\n',ssdatapath)
else
    fprintf('Processing %s ...\n',ssdatapath)
    if nargin>1
        load(infile,'sysc')
        if isa(sysc,'component_mode_model')
            A = sysc.sys.A;
            n = size(A,1);
            h = n/2;
            
            eigenfrequencies = full(sqrt(-diag(A(h+1:n,1:h)))');
            d = full(diag(A(h+1:n,h+1:n)))';
            proportionalDampingVec = -0.5*d./eigenfrequencies;
            m = d==0 & eigenfrequencies==0;
            proportionalDampingVec(m) = Inf;
            eigenfrequencies = eigenfrequencies/2/pi;
            
            Phim = sysc.sys.B(h+1:n,:)';
            Phi  = sysc.sys.C(:,1:h);
            
            FEM_IO.inputs_name = {sysc.inputs.name}';
            idx = {sysc.inputs.indexes};
            if ~all(diff(cell2mat(idx))==1)
                error('Inputs indices do not monotically increase!')
            end
            FEM_IO.inputs_size = cellfun(@numel,idx)';
            FEM_IO.outputs_name = {sysc.outputs.name}';
            idx = {sysc.outputs.indexes};
            if ~all(diff(cell2mat(idx))==1)
                error('Inputs indices do not monotically increase!')
            end
            FEM_IO.outputs_size = cellfun(@numel,idx)';
            
        else
            error('Unrecognized data format!')
        end
    else
        load(infile,'A','B','C','inputTable','outputTable',...
            'eigenfrequencies','proportionalDampingVec')
        n = size(A,1);
        h = n/2;
        
        %{
        eigenfrequencies = full(sqrt(-diag(A(h+1:n,1:h)))');
        d = full(diag(A(h+1:n,h+1:n)))';
        proportionalDampingVec = -0.5*d./eigenfrequencies;
        m = d==0 & eigenfrequencies==0;
        proportionalDampingVec(m) = Inf;
        eigenfrequencies = eigenfrequencies/2/pi;
        %}
        
        Phim = B(h+1:n,:)';
        Phi  = C(:,1:h);
        
        FEM_IO.inputs_name = inputTable.Row;
        FEM_IO.inputs_size = inputTable.size;
        FEM_IO.outputs_name = outputTable.Row;
        FEM_IO.outputs_size = outputTable.size;
        
        idx = inputTable.indices;
        if ~all(diff(cell2mat(idx))==1)
            error('Inputs indices do not monotically increase!')
        end
        idx = outputTable.indices;
        if ~all(diff(cell2mat(idx))==1)
            error('Outputs indices do not monotically increase!')
        end
    end
    
    save(outfile,...
        'eigenfrequencies','proportionalDampingVec',...
        'Phi','Phim','FEM_IO')
end
end