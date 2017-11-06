classdef AppRankPooling < dagnn.ElementWise
  % author: Hakan Bilen
  % dagnn wrapper for approximate rank pooling
  
  properties
    scale = 1 
  end
    
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2}) * obj.scale ;
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2);
      derInputs{1} = vl_nnarpooltemporal(inputs{1},inputs{2},derOutputs{1}) * obj.scale;
      derParams = {} ;
    end
    
    function outputSizes = getOutputSizes(obj, inputSizes)
      % This is not correct, dim(4) depends on inputs{2}
      outputSizes{1} = inputSizes{1} ;
    end
    
    function obj = AppRankPooling(varargin)
      obj.load(varargin) ;  
    end  
    
  end
end

