function [distortion, QL, dataout] = Max_Lloyd(levels, data, meps)

% get the sizes and min, max values of the data
[M,N,L] = size(data);
data_maximum = max(max(max(data)));
data_minimum = min(min(min(data)));

% Initialize arrays for our data
data = double(data);
distance = zeros(1, levels); % Euqlidean error
Chosen_level = zeros(1, M*N); % chosen level per pixel

dataout = zeros(M, N, L); % Quantized data

iterations = 20;
distortion = zeros(1, iterations);

% choose initial representation levels
f = double(data_minimum) + randperm(data_maximum-data_minimum, 3*levels);
f = reshape(f, 3, levels);
QL = f; % updated representation levels


for ii=1:iterations
    
    % Get the specific colors
    R = data(:,:,1);
    R = R(:);
    G = data(:,:,2);
    G = G(:);
    B = data(:,:,3);
    B = B(:);
    
    % for every pixel choose closest level
    for jj=1:(M*N)
        for level = 1:levels
            distance(level) = sqrt((R(jj)-f(1,level))^2 + (G(jj)-f(2,level))^2 + (B(jj)-f(3,level))^2);
        end
        [~, min_ind] = min(distance);
        Chosen_level(jj) = min_ind;
    end
    
    % update QL to new representation levels according to Max-Lloyd
    for level = 1:levels
        ind = find (Chosen_level == level);
        QL(1, level) = round(sum(R(ind)) / length(ind));
        QL(2, level) = round(sum(G(ind)) / length(ind));
        QL(3, level) = round(sum(B(ind)) / length(ind));
    end
    
    % update quantized picture for distortion calculations
    for level = 1:levels
        pixel_ind = find(Chosen_level == level);
        dataout(pixel_ind) = f(1, level);
        dataout(pixel_ind+M*N) = f(2, level);
        dataout(pixel_ind+2*M*N) = f(3, level);
    end
    
    % calculate total MSE
    distortion(ii) = sum(sum(sum((data-dataout).^2)))/(M*N*L);
        
    % calculate normalized error
    if (ii ~= 1)
        error = abs(distortion(ii) - distortion(ii-1)) / distortion(ii-1);
        if (error <= meps)
            break
        end
    end
    
    f = QL;

end

distortion = distortion(distortion~=0);

end