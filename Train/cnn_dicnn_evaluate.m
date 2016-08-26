function info = cnn_dicnn_evaluate(varargin)
%CNN_DICNN Demonstrates evaluation with a trained CNN on UCF101 dataset


run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', 'matconvnet', 'matlab', 'vl_setupnn.m')) ;

addpath Layers Datasets

opts.dataDir = fullfile('data','UCF101') ;
opts.expDir  = fullfile('exp', 'UCF101') ;
opts.modelPath = fullfile('models','net-ucf101-split1.mat');
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 8 ;

opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.ARPoolLayer = 'conv0'; % before conv1

opts.split = 1; % data split
opts.reverseDyn = 0; % reverse video frames e.g.[N:-1:1]
opts.train = struct() ;
opts.train.gpus = [];
opts.train.batchSize = 	 4;
opts.train.numSubBatches = 1;
opts.train.numEpochs = 1;

opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = dagnn.DagNN.loadobj(load(opts.modelPath));
% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath,'file')
  imdb = load(opts.imdbPath) ;
else
  imdb = cnn_ucf101_setup_data('dataDir', opts.dataDir, 'lite', opts.lite) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

% UCF101 has 3 data splits
if opts.split>3
  error('split should be <=3');
end
imdb.images.set = imdb.images.sets(opts.split,:);

% reverse frame order
if opts.reverseDyn
  for i=1:numel(imdb.images.names)
    imdb.images.names{i} = imdb.images.names{i}(end:-1:1);
  end
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
opts.train.train = NaN ;
opts.train.val = find(imdb.images.set==3) ;

[~, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
% bopts.averageImage = []; 
bopts.averageImage = meta.normalization.averageImage ;
% bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;


fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;



% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------

% batch refers to videos (not for frames)
if isempty(batch)
  inputs = {'input', [], 'label', [], 'VideoId1', [], 'VideoId2', []};
  return;
end

isVal = 1 ;

if ~isVal, transformation='stretch'; else transformation='none';end

names = imdb.images.names(batch);


% images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;

namesM = {};
nVids = numel(batch);

VideoId1 = [];
VideoId2 = [];

% step-size
stepSize = 6;
% pool nFrames into a dynamic image
nFrames = 10;
% number of dynamic images to be max pooled later
nDynImgs = 10;


c1 = 1;
for v=1:nVids
  
  name = names{v};
  nFrms = numel(name);

  nSample = nFrames;
  nr = numel(1:stepSize:nFrms);
  
  % jitter by removing 75 % and limit a batch to nMaxs * nSamples images
  if nr > 1 && (~isVal || nr>nDynImgs)
    rat = min(nDynImgs,ceil(0.75*nr));
    ri = randperm(nr);
    ri = ri(1:rat);
    r = zeros(1,nr);
    r(ri) = 1;
  else
    r = ones(1,nr);
  end
  
  c3 = 1;
  c2 = 0;
  
  for f=1:stepSize:nFrms
    if r(c3)
      idx = f:min(f+nSample-1,nFrms) ;
      namesM{end+1} = name(idx);
      VideoId1 = [VideoId1 c1 * ones(1,numel(idx))];
      c1 = c1 + 1;
      c2 = c2 + 1;
    end
    c3 = c3 + 1;
  end
  VideoId2 = [VideoId2 v * ones(1,c2) ] ;
end

images = strcat([imdb.imageDir filesep], horzcat(namesM{:}) ) ;

im = cnn_video_get_batch(images, VideoId1, opts, ...
  'transformation', transformation, 'prefetch', nargout == 0, ...
  'subMean', false) ;

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  inputs = {'input', im, 'label', imdb.images.label(batch), ...
    'VideoId1', VideoId1, 'VideoId2', VideoId2};
end
