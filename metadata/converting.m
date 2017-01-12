%% READ DATA FILE

fid = fopen('U07_CH0.dat','r');
datacell = textscan(fid, '%f');
fclose(fid);
raw_data = datacell{1};


%% CONVERTING DATA
final_conversion_factor = 0.015258789;
physical_value = raw_data ./ 2^16 - 2.5 * final_conversion_factor;
figure(1)
plot(physical_value)
%zero_mean_physical_value = detrend(physical_value);
figure(2)
plot(zero_mean_physical_value)