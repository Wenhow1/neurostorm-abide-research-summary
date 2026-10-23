import torch

def sample_rows(model,seed,ratio):
 # Recreate original block hidden-edge count, then use unbiased integer rounding
 # to bracket it with whole-ROI row/column masks. This matches expectation,
 # not the number of edges in every realized mask.
 gen=torch.Generator().manual_seed(seed)
 block=model.sample_mask(1,device=torch.device('cpu'),mask_ratio=ratio,generator=gen)
 n=116;off=~torch.eye(n,dtype=torch.bool)[None,None];target=int((block&off).sum())
 counts=[n*(n-1)-(n-k)*(n-k-1)for k in range(n+1)]
 hi=next(k for k,v in enumerate(counts)if v>=target);lo=max(0,hi-1)
 prob=(target-counts[lo])/(counts[hi]-counts[lo])if counts[hi]!=counts[lo]else 0
 k=hi if float(torch.rand((),generator=gen))<prob else lo
 selected=torch.zeros(n,dtype=torch.bool);selected[torch.randperm(n,generator=gen)[:k]]=True
 mask=(selected[:,None]|selected[None,:])[None,None]
 return mask,{'ROI_count':k,'hidden_edges_directed':counts[k],'block_target_directed':target,'expected_directed':counts[lo]*(1-prob)+counts[hi]*prob,'hi_probability':prob}
