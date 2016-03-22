classdef TemporalPooling < dagnn.ElementWise
  % author: Hakan Bilen
  % dagnn wrapper for approximate rank pooling
  
  properties
    method = 'max';
  end
 
  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnpooltemporal(inputs{1},inputs{2},obj.method);
    end
    
    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = cell(1,2);
      derInputs{1} = vl_nnpooltemporal(inputs{1},inputs{2},obj.method,derOutputs{1});
      derParams = {} ;
    end
    
    function obj = TemporalPooling(varargin)
      obj.load(varargin) ;  
    end  
    
  end
end

