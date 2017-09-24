classdef AppRankPooling < dagnn.ElementWise
  % author: Hakan Bilen
  % dagnn wrapper for approximate rank pooling
  
  properties
    poolSize = 10 % number of images per dynamic image (this number is not really used in training)
  end
 
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2});
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2);
      derInputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2},derOutputs{1});
      derParams = {} ;
    end
    
    function obj = AppRankPooling(varargin)
      obj.load(varargin) ;  
    end  
    
    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = inputSizes{1} ;
      outputSizes{1}(4) = ceil(inputSizes{1}(4) / obj.poolSize) ;
    end

  end
end

