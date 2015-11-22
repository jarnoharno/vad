function q = g729(in_files,out_files)

files = textread(in_files,'%s');

for i = 1:length(files)
    file = files{i};
    [in_dir,in_base,in_ext] = fileparts(file);
    out_file = fullfile(out_files,strcat(in_base,'.txt'));
    fprintf(2,'g729: %s -> %s\n',file,out_file);
    % read audio
    [y,fs] = audioread(file);
    % init g729 with default settings
    vadpar = InitVADPar();
    % resample with the fastest applicable method
    x = y;
    if vadpar.Fs > fs && mod(vadpar.Fs,fs) == 0
        x = interp(y,vadpar.Fs/fs);
    elseif vadpar.Fs < fs && mod(fs,vadpar.Fs) == 0
        x = decimate(y,fs/vadpar.Fs);
    elseif vadpar.Fs ~= fs
        x = resample(y,vadpar.Fs,fs);
    end
    % zero pad
    numframes = floor((length(x)+vadpar.NF-1)/vadpar.NF);
    z = zeros(numframes*vadpar.NF,1);
    z(1:length(x)) = x;
    % add small value to every frame to prevent bad things from happening
    z(1:vadpar.NF:length(z)) = 1/32767;
    % compute vad sequence
    z_start = 1;
    z_end = vadpar.NF;
    vad_seq = {};
    vad_state = 0;
    for j = 1:numframes
        [v,vadpar] = VAD(z(z_start:z_end),vadpar);
        if v ~= vad_state
            vad_seq{end+1} = (j-1)*vadpar.NF/vadpar.Fs;
        end
        vad_state = v;
        z_start = z_start + vadpar.NF;
        z_end = z_end + vadpar.NF;
    end
    vad_seq{end+1} = vadpar.NF*numframes/vadpar.Fs;
    % write sequence to a file
    vad_arr = cell2mat(vad_seq);
    vad_arr = vad_arr(:);
    dlmwrite(out_file,vad_arr);
end

end
