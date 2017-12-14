function imdb = cnn_hmdb51_of_setup_data(varargin)
% CNN_UCF101_SETUP_DATA Initialize UCF101 - Action Recognition Data Set
% http://crcv.ucf.edu/data/UCF101.php
% this script requires UCF101 downloaded and frames extracted in frames
% folder


opts.dataDir = fullfile('data','HMDB51') ;
opts.lite = false ;
% opts = vl_argparse(opts, varargin) ;

%% ------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------
% find images
imagePath = fullfile(opts.dataDir, 'tvl1_flow', 'u', '*') ;
images = dir(imagePath) ;

videoNames = cell(1,numel(images)) ;
frameNames = cell(1,numel(images)) ;
nrFrames = zeros(1,numel(images)) ;
for i=1:numel(images)
  
  frames = dir(fullfile(opts.dataDir,'tvl1_flow','u',images(i).name,'frame*.jpg')) ;
  framesc = cell(1,numel(frames)) ;
  if ~isempty(numel(frames))
    for j=1:numel(frames)
      framesc{j} = frames(j).name ;
    end
    frameNames{i} = framesc ;
    frameNames{i} = strcat(images(i).name,'/',framesc) ;
    nrFrames(i) = numel(framesc) ;
    videoNames{i} = images(i).name ; 
  end
end

videoNames(nrFrames==0) = [] ;
frameNames(nrFrames==0) = [] ;
% nrFrames(nrFrames==0) = [] ;


frameNamesuv = cell(1,numel(frameNames)) ;
for i=1:numel(frameNames)
  nn = frameNames{i} ;
  nn1 = strcat('u/',nn) ;
  nn2 = strcat('v/',nn) ;
  
  frameNamesuv{i} = cell(1,2*numel(nn1)) ;
  frameNamesuv{i}(1:2:end) = nn1 ;
  frameNamesuv{i}(2:2:end) = nn2 ;
end

% find metadata
% ncls = 51 ;

metaPath = fullfile(opts.dataDir, 'hmdb51_splits', '*.txt') ;

splits = dir(metaPath) ;

cats = cell(1,numel(videoNames)) ;
sets = zeros(3,numel(videoNames)) ;
catNames = cell(1,numel(splits)) ;

for i=1:numel(splits)
  j = strfind(splits(i).name,'_test_') ;
  splitno = str2double(splits(i).name(j+11)) ;
  catNames{i} = splits(i).name(1:j-1) ;
  t = importdata(fullfile(opts.dataDir, 'hmdb51_splits', splits(i).name)) ;
  
  vids = cell(1,numel(t.textdata)) ;
  for k=1:numel(t.textdata)
    vids{k} = t.textdata{k}(1:end-4) ;
  end
  
  [ia,ib] = ismember(vids,videoNames) ;
  assert(all(ia)) ;
  sets(splitno,ib) = t.data' ;
  cats(ib) = repmat(catNames(i),numel(ia),1) ;
end

[cu,~,labels] = unique(cats) ;
sets(sets(:)==2) = 3 ;

imdb.classes.name = cu ;
imdb.images.name = videoNames ;
imdb.images.names = frameNamesuv ;
imdb.images.label = labels' ;
imdb.images.sets = sets ;
imdb.imageDir = fullfile(opts.dataDir, 'tvl1_flow') ;
