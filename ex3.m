%% Question 1.2 %%
clear all; close all; clc;

num = 200;
Circles = zeros(num,num);

% Create the circles image
for ii=5:-1:1
    for jj=-100:1:100
        for kk=-100:1:100
            if (jj^2+kk^2) < (20*ii)^2
                Circles(kk+101,jj+101) = 250-50*(ii-1);
            end
        end
    end
end

figure(1);
subplot(1,2,1); imshow(Circles,[]);
title('Circles Image');

% create the noised circles image
Y = Circles + 10*randn(num,num);
subplot(1,2,2); imshow(Y,[]);
title('Noised Circles Image');

%% Question 1.3 %%
lambda = 0.5;
numIter = 75;

% Estimate X using gradient descent with L2 prior
[~, Err1_L2, Err2_L2] = DenoiseByL2(Y(:),Circles(:),numIter,lambda);

figure(3);
plot(1:numIter,mag2db(Err1_L2),'b'); hold on;
plot(1:numIter,mag2db(Err2_L2),'r');
title('Errors of Gradient Descent - L2');
xlabel('iterations'); ylabel('Error Value');
legend('Error 1', 'Error 2');

%% Question 1.5 %%
lambda = 15;
numIter = 200;

% Estimate X using gradient descent with TV prior
[~, Err1_TV, Err2_TV] = DenoiseByTV(Y(:),Circles(:),numIter,lambda);

figure(4);
plot(1:numIter,mag2db(Err1_TV),'b'); hold on;
plot(1:numIter,mag2db(Err2_TV),'r');
title('Errors of Gradient Descent - TV');
xlabel('iterations'); ylabel('Error Value');
legend('Error 1', 'Error 2');

%% Question 2.1 %%
close all; clear all; clc;
load('..\rabbit.mat');
implay(rabbit);

%% Question 2.3 %%
meps = 0.02;
levels = 9;

[m,n,l,z] = size(rabbit);
Appended_pic = zeros(m*z,n,l);

for ii=1:z
    Appended_pic(((ii-1)*m+1):(ii*m),:,:) = rabbit(:,:,:,ii);
end

[distortion, QL, dataout] = Max_Lloyd(levels, Appended_pic, meps);
distortion = distortion(distortion~=0);

figure(1);
plot(1:length(distortion),distortion);
title('Distortion values as a function of iterations');
xlabel('iteration number');
ylabel('Distortion');

%% Question 2.4 %%
iterations = 20;
Distortions_Divisions = zeros(iterations,z-1);
Best_Distortions_Divisions = zeros(1,z-1);
Iterations_Divisions = zeros(1,z);

for division=1:(z-1)
    division
    Appended_pic_1 = zeros(division*m,n,l);
    Appended_pic_2 = zeros((z-division)*m,n,l);

    for ii=1:division
        Appended_pic_1(((ii-1)*m+1):(ii*m),:,:) = rabbit(:,:,:,ii);
    end
    for ii=(division+1):z
        Appended_pic_2(((ii-division-1)*m+1):((ii-division)*m),:,:) = rabbit(:,:,:,ii);
    end
    
    [distortion_1, QL_1, dataout_1] = Max_Lloyd(levels, Appended_pic_1, meps);
    [distortion_2, QL_2, dataout_2] = Max_Lloyd(levels, Appended_pic_2, meps);

    distortion_1_squeezed = distortion_1(distortion_1~=0);
    distortion_2_squeezed = distortion_2(distortion_2~=0);
    
    Iterations_Divisions(division) = min(length(distortion_1_squeezed), length(distortion_2_squeezed));
    Distortions_Divisions(:,division) = distortion_1*(division/z)+distortion_2*(1-division/z);
    Best_Distortions_Divisions(division) = distortion_1_squeezed(end)*(division/z)+distortion_2_squeezed(end)*(1-division/z);
end

[val, ind] = min(Distortions_Divisions(end,:));

figure(2);
subplot(2,1,1); stem(1:(z-1),Best_Distortions_Divisions);
title('Distortion as a function of Video Division');
xlabel('division index'); ylabel('distortion');

subplot(2,1,2); stem(1:Iterations_Divisions(ind),Distortions_Divisions(1:Iterations_Divisions(ind),ind));
title('Distortion as a function of Iteration for the best division');
xlabel('iteration'); ylabel('distortion');

%% Question 2.5 %%
chosen_division = ind;
Q_rabbit = zeros(m,n,l,z);
Q_rabbit_merged = zeros(m,n,l,z);

for ii=1:1:z
    Q_rabbit(:,:,:,ii) = dataout(((ii-1)*m+1):(ii*m),:,:);
    
    if ii<=division
        Q_rabbit_merged(:,:,:,ii) = dataout_1(((ii-1)*m+1):(ii*m),:,:);
    else
        Q_rabbit_merged(:,:,:,ii) = dataout_2(((ii-division-1)*m+1):((ii-division)*m),:,:);
    end
end

implay(uint8(Q_rabbit));
implay(uint8(Q_rabbit_merged));

%% Question 3.1 %%
close all; clear all; clc;
% generating data
t = (0:0.01:1).';
s = (0:0.01:1).';
y = (t+2) + (s+3) + 0.2*randn(size(t,1),1);
X = [t s y];

figure(1);
scatter3(X(:,1), X(:,2), X(:,3), 'b'); 
title('Data');
xlabel('t');
ylabel('s');
zlabel('y');
%% Question 3.2 %%
mean = mean(X);
std = std(X);
X_Processed = X - repmat(mean,length(X),1);
X_Processed = X_Processed ./ repmat(std,length(X),1);
%% Question 3.3 %%
X_cov = cov(X_Processed);
[~,S,V] = svd(X_cov); % using svd as recommanded
X_cov
V
S = diag(S)

% building trasnform matrix
Transform = V(:,1); % vector with largest eigenvalue

%% Question 3.4 %%
X_Transformed =  X_Processed * Transform;

%% Question 3.5 %%
iTransform = Transform';
X_Reconstructed = X_Transformed * iTransform;

X_Reconstructed = X_Reconstructed .* repmat(std,length(X),1);
X_Reconstructed = X_Reconstructed + repmat(mean,length(X),1);

figure(2);
scatter3(X(:,1), X(:,2), X(:,3), 'b'); hold on;
scatter3(X_Reconstructed(:,1), X_Reconstructed(:,2), X_Reconstructed(:,3), 'r');
legend ('X original','X Reconstructed');
title('Data');
xlabel('t');
ylabel('s');
zlabel('y');

[m , n] = size(X_Reconstructed);
MSE_Q5 = sum(sum((X-X_Reconstructed).^2)) / (m*n)

%% Question 3.6 - part II %%
close all; clear all; clc;
load ('../faces')
NumOfFaces = 25;

figure(3); 
suptitle('Faces');
for ii = 1 : NumOfFaces
    face = reshape(X(ii,:),32,32);
    figure(3); subplot(5,5,ii); 
    imshow(face,[]);
end
%% Question 3.7 %%
Mean_Faces = reshape(mean(X),32,32);
figure(4); 
suptitle('Mean All Faces');
imshow(Mean_Faces,[]);

%% Question 3.8 %%

% PCA
mean = mean(X);
std = std(X);
X_Processed = X - repmat(mean,length(X),1);
X_Processed = X_Processed ./ repmat(std,length(X),1);
[m , n] = size(X);

X_cov = cov(X_Processed);
[~,S,V] = svd(X_cov); % using svd as recommanded
S = diag(S);

TransformDim = [1,2,5,10,25,50,100,200,300,400,500];

MSE = zeros(length(TransformDim),1);
VAR = zeros(length(TransformDim),1);

for ii = 1:length(TransformDim)
    Transform = V(:,1:TransformDim(ii));
    X_Transformed = X_Processed * Transform;
    iTransform = Transform';
    X_Reconstructed = X_Transformed * iTransform;
    X_Reconstructed = X_Reconstructed .* repmat(std,length(X),1);
    X_Reconstructed = X_Reconstructed + repmat(mean,length(X),1);
    MSE(ii) = sum(sum((X-X_Reconstructed).^2)) / (m*n);
    VAR(ii) = sum(S(1:TransformDim(ii))) / sum(S);
end

figure(5);
subplot(1,2,1);
plot(1:1:length(TransformDim),MSE);
title('MSE as a function of Transform Dimension');
xlabel('Transform Dimension');
ylabel('MSE');
subplot(1,2,2);
plot(1:1:length(TransformDim),VAR);
title('VAR as a function of Transform Dimension');
xlabel('Transform Dimension');
ylabel('VAR');

%% Question 3.9 %%
K = [10,50,100,300,500];


for ii = 1:length(K)
    Transform = V(:,1:K(ii));
    X_Transformed = X_Processed * Transform;
    iTransform = Transform';
    X_Reconstructed = X_Transformed * iTransform;
    X_Reconstructed = X_Reconstructed .* repmat(std,length(X),1);
    X_Reconstructed = X_Reconstructed + repmat(mean,length(X),1);
    MSE(ii) = sum(sum((X-X_Reconstructed).^2)) / (m*n);
    VAR(ii) = sum(S(1:K(ii))) / sum(S);
    figure;
    suptitle(['K = ', num2str(K(ii))]);
    for jj = 1:NumOfFaces
        face = reshape(X_Reconstructed(jj,:),32,32);
        subplot(5,5,jj); 
        imshow(face,[]);
    end
end
%% Question 3.10 %%
figure;
suptitle('Eigenface');
for ii = 1:36
    face = reshape(V(:,ii),32,32);
    subplot(6,6,ii); 
    imshow(face,[])
end