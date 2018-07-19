function args = liftprm(prm_src)
if isstruct(prm_src)
    args = prm_src;
elseif ischar(prm_src)
    [~,~,ext] = fileparts(prm_src);
    switch ext
        case '.ubj'
            args = loadubjson(prm_src,'simplifyCell',1);
        case '.json'
            args = loadjson(prm_src,'simplifyCell',1);
        otherwise
            error('simceo:loadprm:file_error','Unrecognized file type! Valid file extensions are ubj or json!')
    end
else
    error('simceo:loadprm:type_error','Input must be either a structure or a filename!')
end
