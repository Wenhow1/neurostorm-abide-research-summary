import math
class Plateau:
 def __init__(self,optimizer):self.optimizer=optimizer;self.best=math.inf;self.bad=0;self.cooldown=0;self.events=0;self.calls=0
 def step(self,value):
  assert math.isfinite(value);self.calls+=1
  improved=value<self.best*.999
  if improved:self.best=value;self.bad=0
  elif self.cooldown==0:self.bad+=1
  if self.cooldown>0:self.cooldown-=1;self.bad=0;return
  if self.bad>=5:
   for g in self.optimizer.param_groups:g['lr']=max(g['lr']*.5,1e-6)
   self.events+=1;self.bad=0;self.cooldown=1
 def state_dict(self):return {k:getattr(self,k) for k in ['best','bad','cooldown','events','calls']}
 def load_state_dict(self,state):
  for k,v in state.items():setattr(self,k,v)
