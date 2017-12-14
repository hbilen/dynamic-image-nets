function imdb = cnn_ucf101_of_setup_data(varargin)
% CNN_UCF101_SETUP_DATA Initialize UCF101 - Action Recognition Data Set
% http://crcv.ucf.edu/data/UCF101.php
% this script requires UCF101 downloaded and frames extracted in frames
% folder

opts.dataDir = fullfile('data','UCF101') ;
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

%% ------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

% find metadata
metaPath = fullfile(opts.dataDir, 'ucfTrainTestlist/classInd.txt') ;

fprintf('using metadata %s\n', metaPath) ;
tmp = importdata(metaPath);
nCls = numel(tmp);

if nCls ~= 101
  error('Wrong meta file %s',metaPath);
end

cats = cell(1,nCls);
for i=1:numel(tmp)
  t = strsplit(tmp{i});
  cats{i} = t{2};
end

imdb.classes.name = sort(cats) ;
imdb.imageDir = fullfile(opts.dataDir, 'tvl1_flow', 'u') ;

%% ------------------------------------------------------------------------
%                                              load image names and labels
% -------------------------------------------------------------------------

fprintf('searching training images ...\n') ;
names = {} ;
name = {};
labels = {} ;
for d = dir(fullfile(imdb.imageDir, 'v_*'))'
  [~,lab] = ismember(lower(d.name(3:end-8)), lower(cats)) ;
  if lab==0
    error('no class label found for %s',d.name);
  end
  ims = dir(fullfile(imdb.imageDir, d.name, '*.jpg')) ;
  name{end+1} = d.name;
  names{end+1} = strcat([d.name, filesep], {ims.name}) ;
  labels{end+1} = lab ;
  if mod(numel(names), 10) == 0, fprintf('.') ; end
  if mod(numel(names), 500) == 0, fprintf('\n') ; end
  %fprintf('found %s with %d images\n', d.name, numel(ims)) ;
end
% names = horzcat(names{:}) ;

labels = horzcat(labels{:}) ;
% labels = [labels ; labels] ;
labels = labels(:)' ;

for i=1:numel(names)
  nn = names{i} ;
  nn1 = strcat('u/',nn) ;
  nn2 = strcat('v/',nn) ;
  
  names{i} = cell(1,2*numel(nn1)) ;
  names{i}(1:2:end) = nn1 ;
  names{i}(2:2:end) = nn2 ;
end

imdb.images.id = 1:numel(names) ;
imdb.images.name = name ;
imdb.images.names = names ;
imdb.images.label = labels ;
imdb.imageDir = fullfile(opts.dataDir, 'tvl1_flow') ;

%% ------------------------------------------------------------------------
%                                                 load train / test splits
% -------------------------------------------------------------------------

fprintf('labeling data...(this may take couple of minutes)') ;
imdb.images.sets = zeros(3, numel(names)) ;
setNames = {'train','test'};
setVal = [1,3];

for s=1:numel(setNames)
  for i=1:3
    trainFl = fullfile(opts.dataDir, 'ucfTrainTestlist',sprintf('%slist%02d.txt',...
      setNames{s},i)) ;
    trainList = importdata(trainFl);
    if isfield(trainList,'textdata')
      trainList = trainList.textdata;
    end
    for j=1:numel(trainList)
      tmp = strsplit(trainList{j},'/');
      [~,lab] = ismember(lower(tmp{2}(1:end-4)), lower(name)) ;
      if lab==0
%         error('cannot find the video %s',tmp{2}(1:end-4));
        warning('cannot find the video %s',tmp{2}(1:end-4));
        continue ;
      end
%       if trainList.data(j) ~= labels(lab)
%         error('Labels do not match for %s',tmp{2});
%       end
      imdb.images.sets(i,lab) = setVal(s);
    end
  end  
end
fprintf('\n') ;
%% ------------------------------------------------------------------------
%                                                            Postprocessing
% -------------------------------------------------------------------------

% sort categories by WNID (to be compatible with other implementations)
[imdb.classes.name,perm] = sort(imdb.classes.name) ;
relabel(perm) = 1:numel(imdb.classes.name) ;
ok = imdb.images.label >  0 ;
imdb.images.label(ok) = relabel(imdb.images.label(ok)) ;

if opts.lite
  % pick a small number of images for the first 10 classes
  % this cannot be done for test as we do not have test labels
  clear keep ;
  for i=1:10
    sel = find(imdb.images.label == i) ;
    train = sel(imdb.images.sets(1,sel) == 1) ;
    test = sel(imdb.images.sets(1,sel) == 3) ;
    keep{i} = [train test] ;
  end
  keep = keep{:};
  imdb.images.id = imdb.images.id(keep) ;
  imdb.images.name = imdb.images.name(keep) ;
  imdb.images.names = imdb.images.names(keep) ;
  imdb.images.sets = imdb.images.sets(1,keep) ;
  imdb.images.label = imdb.images.label(keep) ;
end
