classdef ErrorMultiClass < dagnn.Loss
% author: Hakan Bilen
% computes multi-class accuracy
% inputs{1}->scores
% inputs{2}->gt labels
  properties
    nImgPerClass = []
    nCorPred = []
    accuracy = []
    resetLayer = false 
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      
      if numel(inputs)~=2
        error('wrong number of inputs');
      end
      
      nCls = size(inputs{1},3);
      
      if obj.resetLayer || isempty(obj.nImgPerClass)
        obj.nImgPerClass = zeros(1,size(inputs{1},3));
        obj.nCorPred = zeros(1,size(inputs{1},3));
        obj.accuracy = zeros(1,size(inputs{1},3));
        
        if obj.resetLayer
          obj.resetLayer = false ;
          obj.average = 0 ;
        end
      end
      
      
      [~,predictions] = max(gather(squeeze(inputs{1})),[],1);
      
      for c=1:nCls
        obj.nImgPerClass(c) = obj.nImgPerClass(c) + sum(inputs{2}==c);
        obj.nCorPred(c)     = obj.nCorPred(c) + sum(predictions==c & inputs{2}==c);
      end
      
      ni = obj.nImgPerClass;
      ni(ni==0) = 1;
      
      obj.accuracy = obj.nCorPred ./ ni;
      obj.average = (1-mean(obj.accuracy));
      outputs{1} =  obj.average;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2);
      derParams = {} ;
    end
    
    function reset(obj)
      obj.resetLayer = true ;
%       obj.nImgPerClass = [];
%       obj.nCorPred = [];
%       obj.accuracy = [];
%       obj.average = 0;
    end
    
    
    function obj = ErrorMultiClass(varargin)
      obj.load(varargin) ;
      obj.loss = 'error_multi_class' ;
    end
  end
end
