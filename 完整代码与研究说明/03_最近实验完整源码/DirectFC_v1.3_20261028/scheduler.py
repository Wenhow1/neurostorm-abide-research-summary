"""Pre-registered five-bad-epoch Plateau; next-epoch LR semantics."""
import math
class Plateau:
 def __init__(self,lr=1e-4):self.best=None;self.bad=0;self.cooldown=0;self.lr=lr
 def step(self,bce):
  assert math.isfinite(bce),'nonfinite validation BCE';old=self.lr
  if self.best is None:self.best=bce;return {'reason':'initialize','old_lr':old,'next_lr':self.lr}
  better=bce<self.best*.999
  if better:self.best=bce
  if self.cooldown:self.cooldown-=1;self.bad=0;reason='cooldown'
  elif better:self.bad=0;reason='improved'
  else:
   self.bad+=1;reason='bad_epoch'
   if self.bad==5:self.lr=max(self.lr*.5,1e-6);self.bad=0;self.cooldown=1;reason='reduce' if self.lr<old else 'minimum_lr'
  return {'reason':reason,'old_lr':old,'next_lr':self.lr,'best':self.best,'bad':self.bad,'cooldown':self.cooldown}
 def state_dict(self):return dict(self.__dict__)
 def load_state_dict(self,s):self.__dict__.update(s)
