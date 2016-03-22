classdef L2Normalize < dagnn.ElementWise
  % author: Hakan Bilen
  % dagnn wrapper for l2 normalization
  
  properties
    scale = 1;
    clip = [-inf inf];
    offset = 0;
  end
  
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnl2norm(inputs{1},[obj.scale obj.clip obj.offset]);
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = vl_nnl2norm(inputs{1},[obj.scale obj.clip obj.offset],derOutputs{1});
      derParams = {} ;
    end
    
    function obj = L2Normalize(varargin)
      obj.load(varargin) ;  
    end  
    
  end
end

