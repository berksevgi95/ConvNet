function Main(varargin)

imdb = load('data-set.mat') ;

% Convolutional Neural Network initialization
net = initializeCNN() ;

% Some properties about learning operation are defined 
trainOpts.batchSize = 50 ;
trainOpts.numEpochs = 30 ;
trainOpts.continue = true ;
trainOpts.gpus = [] ;
trainOpts.learningRate = 0.001 ;
trainOpts.expDir = 'data_set/epoches' ;
trainOpts = vl_argparse(trainOpts, varargin);

% Training of CNN
net = cnn_train(net, imdb, @getBatch, trainOpts) ;
net.layers(end) = [] ;

% Test Phase
input_image = imread('img_0423thumb.jpg');
im = im2single(rgb2gray(imresize(input_image,[32,32])));
res = vl_simplenn(net, im) ;


for i=1:size(res(end).x,2)
  [score,pred] = max(squeeze(res(end).x(1,i,:))) ;
end

fileID = fopen('costs.txt','r');
all_costs = fscanf(fileID,'%d');
for i = 1:length(all_costs)
    if i == pred;
        cost = all_costs(i);
    end
end
fclose(fileID);

result_text = 'Predicted cost : ~';
result_text = strcat(result_text,int2str(cost));
result_text = strcat(result_text,'$');
result = insertText(input_image, [0 0 ], result_text);
figure(2) ,imshow(result);

function [im, labels] = getBatch(imdb, batch)

im = imdb.images.data(:,:,batch) ;
im = 256 * reshape(im, 32, 32, 1, []) ;
labels = imdb.images.label(1,batch) ;

