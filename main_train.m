model = 'resnext50' ; % {'cafferef','resnext50','resnext101'}
input = 'rgb' ; % {'rgb','of'}
dataset = 'ucf101' ; % {'ucf101','hmdb51'}  hmdb51 requires more iterations to train (add more epochs to learning rate)
opts.train.batchSize = 128 ;
opts.train.numSubBatches = 32 ; % increase the number (16,32) if it does not fit into gpu mem 
opts.epochFactor = 5 ;
opts.split = 1 ;

opts.train.gpus = 1 ;

run matconvnet/matlab/vl_setupnn.m ;
vl_contrib install mcnExtraLayers ; vl_contrib setup mcnExtraLayers ;
vl_contrib install autonn ; vl_contrib setup autonn ;

% addpath(fullfile('matconvnet','contrib','mcnExtraLayers','matlab')) ;

opts.expDir = ['exp/' model 'rgb-arpool-split' num2str(opts.split)] ;
if strcmp(input,'rgb')  
  opts.DropOutRate = 0.5 ;
  trainfn = @cnn_dicnn_rgb ;
elseif strcmp(input,'of')  
  opts.DropOutRate = 0.8 ;
  trainfn = @cnn_dicnn_of ;
end

if strcmp(model,'cafferef')  

  opts.pool1Layer = 'conv1' ;
  % download from http://www.vlfeat.org/matconvnet/models/imagenet-caffe-ref.mat
  opts.modelPath = fullfile('models','imagenet-caffe-ref.mat') ;
  opts.networkFn = @cnn_init_cafferef ;
  
  if strcmp(input,'rgb')  
    opts.train.learningRate = 1e-3 * [ones(1,2) 0.1*ones(1,2)] ;
  else
    opts.train.learningRate = 3e-3 * [ones(1,10) 0.1*ones(1,2)] ;
  end

  opts.train.numEpochs = numel(opts.train.learningRate) ;
elseif strcmp(model,'resnext50') || strcmp(model,'resnext101')
  % download from http://www.robots.ox.ac.uk/~albanie/models/pytorch-imports/resnext_50_32x4d-pt-mcn.mat
  % download from http://www.robots.ox.ac.uk/~albanie/models/pytorch-imports/resnext_101_32x4d-pt-mcn.mat
  if strcmp(model,'resnext50')
    opts.modelPath = fullfile('models','resnext_50_32x4d-pt-mcn.mat') ;
  else
    opts.modelPath = fullfile('models','resnext_101_32x4d-pt-mcn.mat') ;
  end
  opts.modelPath = fullfile('models','resnext_50_32x4d-pt-mcn.mat') ;
  opts.networkFn = @cnn_init_resnext ;
  if strcmp(input,'rgb')  
    opts.train.learningRate = 1e-2 * [ones(1,2) 0.1*ones(1,8) ] ;
  else
    opts.train.learningRate = 1e-2 * [ones(1,2) 0.1*ones(1,2) ] ;
  end
end

addpath dicnn ;

[net, info] = trainfn(opts)
