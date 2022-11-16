classdef MyMsgObj
   properties
      Time
      Topic
      MessageType
      FileOffset
      Data
   end
   methods
      function obj = MyMsgObj(msgPropsTable, msgData)
         if nargin > 0
            obj.Time = msgPropsTable{1,1};
            obj.Topic = char(msgPropsTable{1,2});  % table contains categorical object, we want string only
            obj.MessageType = char(msgPropsTable{1,3});
            obj.FileOffset = msgPropsTable{1,4};
            obj.Data = msgData;
         end
      end
   end
end