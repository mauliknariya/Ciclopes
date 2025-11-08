import numpy as np
import pandas as pd
from math import log10, floor, ceil
from scipy import sparse
from scipy.stats import zscore
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
import torch
import torch.nn.functional as F
import anndata
import scanpy
import warnings
warnings.filterwarnings('ignore')


def load_sanity(sanity_dir):
    gex_file = f'{sanity_dir}/gex_ltq.txt'
    dfgex = pd.read_csv(gex_file, index_col=0, sep='\t')
    dfgex = dfgex.sort_index(axis=0).sort_index(axis=1).copy()
    adata = anndata.AnnData(obs=pd.DataFrame(index=dfgex.columns), var=pd.DataFrame(index=dfgex.index))
    adata.layers['gene_expression'] = dfgex.T
    adata.var_names_make_unique()
    return adata

    
def load_sanity_unsp(sanity_dir, velocyto_file=None, deepcycle_file=None):
    # Sanity data
    gex_file = f'{sanity_dir}/gex_ltq.txt'
    unspliced_file = f'{sanity_dir}/unspliced_ltq.txt'
    spliced_file = f'{sanity_dir}/spliced_ltq.txt'
    dfgex = pd.read_csv(gex_file, index_col=0, sep='\t')
    dfun = pd.read_csv(unspliced_file, index_col=0, sep='\t')
    dfsp = pd.read_csv(spliced_file, index_col=0, sep='\t')
    
    common_genes = set(dfgex.index).intersection(set(dfun.index).intersection(set(dfsp.index)))
    dfgex = dfgex[dfgex.index.isin(common_genes)].copy()
    dfun = dfun[dfun.index.isin(common_genes)].copy()
    dfsp = dfsp[dfsp.index.isin(common_genes)].copy()

    dfgex = dfgex.sort_index(axis=0).sort_index(axis=1).copy()
    dfun = dfun.sort_index(axis=0).sort_index(axis=1).copy()
    dfsp = dfsp.sort_index(axis=0).sort_index(axis=1).copy()

    adata = anndata.AnnData(obs=pd.DataFrame(index=dfgex.columns), var=pd.DataFrame(index=dfgex.index))
    adata.layers['gene_expression'] = dfgex.T
    adata.layers['unspliced'] = dfun.T
    adata.layers ['spliced'] = dfsp.T
    adata.var_names_make_unique()
    
    # velocyto
    if velocyto_file is not None:
        adata_velo = anndata.read_loom(velocyto_file)
        adata_velo.var_names_make_unique()
        adata_velo.obs.index = [x.split(':')[-1] for x in adata_velo.obs.index]
        sorted_obs = adata_velo.obs.index.sort_values()
        sorted_var = adata_velo.var.index.sort_values()
        adata_velo = adata_velo[sorted_obs, sorted_var].copy()
        cells = list(set(adata_velo.obs.index.tolist()).intersection(adata.obs.index))
        genes = list(set(adata_velo.var.index.tolist()).intersection(adata.var.index))
        adata = adata[adata.obs.index.isin(cells), adata.var.index.isin(genes)].copy()
        adata_velo = adata_velo[adata_velo.obs.index.isin(cells), adata_velo.var.index.isin(genes)].copy()
        dfncts = pd.DataFrame(data=adata_velo.layers['matrix'].toarray().sum(axis=1), 
                              dtype=np.int64, index=adata_velo.obs.index, columns=['n_counts'])
        dfmtx = pd.DataFrame(data=adata_velo.layers['matrix'].toarray(), 
                             dtype=np.int64, index=adata_velo.obs.index, columns=adata_velo.var.index)
        adata.layers['matrix'] = dfmtx
        adata.obs = pd.concat([adata.obs, dfncts], axis=1)
    
    # Deepcycle
    if deepcycle_file is not None:
        adata_deep = anndata.read_h5ad(deepcycle_file)
        dftheta = pd.DataFrame(adata_deep.obs['cell_cycle_theta'])
        dftheta.index = [x.split(':')[-1] for x in dftheta.index]
        dftheta.columns = ['theta']
        cells = list(set(adata.obs.index).intersection(set(dftheta.index)))
        adata = adata[adata.obs.index.isin(cells)]
        dftheta = dftheta[dftheta.index.isin(cells)]
        adata.obs = pd.concat([adata.obs, dftheta], axis=1)    
    
    return adata


def kernel_smooth(x, y, fwhm=None, num_pts=10):
    '''
    Performs kernel smoothing using a Gaussian kernel
    x: x-values
        1D array, dtype=float
    y: y-values
        1D array, dtype=float
    fwhm: full width at half maximum (bandwidth)
         if None, then will be equal to (xmax-xmin) / 100
        float
    '''
    x, y = zip(*sorted(zip(x, y)))
    x = np.asarray(x)
    y = np.asarray(y)
    if fwhm is None:
        fwhm = 0.1*(max(x) - min(x))
    if num_pts is None:
        num_pts = int((max(x) - min(x)) / fwhm)
    delx = (max(x) - min(x)) / num_pts          
    
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    xsm = np.linspace(min(x), max(x), num_pts)
    sigma = fwhm2sigma(fwhm)
    dist = cdist(xsm.reshape(len(xsm), 1), x.reshape(len(x), 1))
    kernel = np.exp(-dist**2/sigma**2)
    norm_kernel = normalize(kernel, axis=1, norm='l1')
    ysm = np.dot(norm_kernel, y)
    
    return xsm, ysm


def kernel_smooth_periodic(x, y, fwhm=None, num_pts=None):
    '''
    Performs kernel smoothing using a periodic Gaussian (von Mises-Fisher) kernel
    x: x-values
        1D array, dtype=float
    y: y-values
        1D array, dtype=float
    fwhm: full width at half maximum (bandwidth)
         if None, then will be equal to (xmax-xmin) / 100
        float
    '''
    x, y = zip(*sorted(zip(x, y)))
    x = np.asarray(x)
    y = np.asarray(y)
    if fwhm is None:
        fwhm = 2.0*np.pi*0.1*(max(x) - min(x))
    if num_pts is None:
        num_pts = int((max(x) - min(x)) / fwhm)
    delx = (max(x) - min(x)) / num_pts          
    
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    xsm = np.linspace(min(x), max(x), num_pts)
    sigma = fwhm2sigma(fwhm)
    dist = -2.0*np.pi*cdist(xsm.reshape(len(xsm), 1), x.reshape(len(x), 1))
    kernel = np.exp(np.cos(dist)/(sigma**2))
    norm_kernel = normalize(kernel, axis=1, norm='l1')
    ysm = np.dot(norm_kernel, y)
    return xsm, ysm

def kernel_smooth_periodic_torch(x, Y, fwhm=None, num_pts=None, device='cuda'):
    x = torch.tensor(x, dtype=torch.float32, device=device)
    Y = torch.tensor(Y, dtype=torch.float32, device=device) 
    xsorted, idxs = x.sort(dim=0)
    Ysorted = Y[idxs]
    if fwhm is None:
        fwhm = torch.tensor(2.0*np.pi*0.01*(max(x) - min(x)), dtype=torch.float32, device=device)
    if num_pts is None:
        num_pts = torch.tensor(((max(x) - min(x)) / fwhm), dtype=torch.int32, device=device)
    delx = torch.tensor((max(x) - min(x)) / num_pts, dtype=torch.float32, device=device)          
    
    def fwhm2sigma(fwhm):
        return fwhm / np.sqrt(8 * np.log(2))

    xsm = torch.linspace(min(x), max(x), num_pts, device=device)
    sigma = fwhm2sigma(fwhm)
    dist = torch.cdist(xsm.unsqueeze(1), xsorted.unsqueeze(1), p=2)
    kernel = torch.exp(-dist**2/sigma**2)
    norm_kernel = F.normalize(kernel, p=1, dim=1, eps=1e-12)
    Ysm = norm_kernel @ Ysorted
    return xsm, Ysm

    
def explained_variance(theta, gex, fwhm=None, num_pts=None):
    '''
    Calculates the explained variance in data for oscillatory gene expression dynamics modeled
    using a periodic gaussian kernel smoothing.
    '''
    tsm, gsm = kernel_smooth_periodic(x=theta, y=gex, fwhm=fwhm, num_pts=num_pts)
    gex_model = np.asarray([gsm[np.argmin(abs(tsm-t))] for t in theta])
    var_model = np.sum(np.square(gex - gex_model))
    var_data = np.sum(np.square(gex - np.mean(gsm)))
    if var_data==0:
        expvar = 0.0
    else:
        expvar = 1.0 - var_model / var_data
    
    ampFC = (np.max(gex_model) - min(gex_model)) / np.mean(gex_model)
    return expvar, ampFC

def explained_variance_torch(theta, GEX, fwhm=None, num_pts=None, device='cuda'):
    theta = torch.tensor(theta, dtype=torch.float32, device=device)
    GEX = torch.tensor(GEX, dtype=torch.float32, device=device) 
    thetasm, GEXsm = kernel_smooth_periodic_torch(x=theta,Y=GEX, fwhm=fwhm, num_pts=num_pts)
    diff = torch.abs(thetasm.unsqueeze(1) - theta.unsqueeze(0))
    mask = torch.argmin(diff, dim=0)
    GEXmodel = GEXsm[mask]
    GEXmean = torch.mean(GEX, axis=0)
    varmodel = torch.square(GEX - GEXmodel).sum(axis=0)
    varGEX = torch.square(GEX - GEXmean).sum(axis=0) + 1e-6
    ExpVar = 1.0 - varmodel / varGEX
    GEXmax = torch.max(GEXsm, axis=0).values
    GEXmin = torch.min(GEXsm, axis=0).values
    AmpFC = (GEXmax - GEXmin) / GEXmean
    return ExpVar, AmpFC


def sg2m_score(adata, organism):
    # Sgenes
    s_genes = np.array(['Atad2', 'Blm', 'Brip1', 'Casp8ap2', 'Ccne2', 'Cdc45', 'Cdc6',
       'Cdca7', 'Chaf1b', 'Clspn', 'Dscc1', 'Dtl', 'E2f8', 'Exo1', 'Fen1',
       'Gins2', 'Gmnn', 'Hells', 'Mcm2', 'Mcm4', 'Mcm5', 'Mcm6', 'Mlf1ip',
       'Msh2', 'Nasp', 'Pcna', 'Pola1', 'Pold3', 'Prim1', 'Rad51',
       'Rad51ap1', 'Rfc2', 'Rpa2', 'Rrm1', 'Rrm2', 'Slbp', 'Tipin',
       'Tyms', 'Ubr7', 'Uhrf1', 'Ung', 'Usp1', 'Wdr76'])
    # G2M genes
    g2m_genes = np.array(['Anln', 'Anp32e', 'Aurka', 'Aurkb', 'Birc5', 'Bub1', 'Cbx5',
       'Ccnb2', 'Cdc20', 'Cdc25c', 'Cdca2', 'Cdca3', 'Cdca8', 'Cdk1',
       'Cenpa', 'Cenpe', 'Cenpf', 'Ckap2', 'Ckap2l', 'Ckap5', 'Cks1b',
       'Cks2', 'Ctcf', 'Dlgap5', 'Ect2', 'Fam64a', 'G2e3', 'Gas2l3',
       'Gtse1', 'Hjurp', 'Hmgb2', 'Hmmr', 'Hn1', 'Kif11', 'Kif20b',
       'Kif23', 'Kif2c', 'Lbr', 'Mki67', 'Ncapd2', 'Ndc80', 'Nek2',
       'Nuf2', 'Nusap1', 'Psrc1', 'Rangap1', 'Smc4', 'Tacc3', 'Tmpo',
       'Top2a', 'Tpx2', 'Ttk', 'Tubb4b', 'Ube2c'])
    if organism=='human':
        s_genes = [x.upper() for x in s_genes]
        g2m_genes = [x.upper() for x in g2m_genes]
    elif organism=='fly':
        s_genes = [x.lower() for x in s_genes]
        g2m_genes = [x.lower() for x in g2m_genes]
    
    # Load data and calculate S-G2M scores
    scanpy.tl.score_genes_cell_cycle(adata=adata, s_genes=s_genes, g2m_genes=g2m_genes)
    adata.obs['S_zscore'] = zscore(adata.obs['S_score'].values)
    adata.obs['G2M_zscore'] = zscore(adata.obs['G2M_score'].values)
    return adata
    
def to_polar(x, y, eps=1e-6):
    '''
    converts a point could (in 2D) from cartesian (x, y)-> polar(r, θ) polar cordinates
    assumes the center of the point could to be the origin the polar cordinates
    '''
    x = np.asarray(x)
    y = np.asarray(y)
    x0, y0 = np.mean(x), np.mean(y)
    dx, dy = x - x0, y - y0
    r = np.hypot(dx + eps, dy + eps)                 
    phi = (np.pi + np.arctan2(dy + eps, dx + eps)) / (2*np.pi)
    return r, phi


def to_cartesian(r, phi, center):
    cx, cy = center
    x = cx + r * np.cos(2*np.pi*phi)
    y = cy + r * np.sin(2*np.pi*phi)
    return -x, -y
    

def estimate_theta(x, y, fwhm=0.5, num_pts=100):
    r, phi = to_polar(x, y)
    psm, rsm = kernel_smooth_periodic(x=phi, y=r, fwhm=fwhm, num_pts=num_pts)
    xsm, ysm = to_cartesian(rsm, psm, (np.mean(x), np.mean(y)))
    arclength_tot = np.sum(np.sqrt(np.diff(xsm)**2 + np.diff(ysm)**2)) 
    
    def dist_loop(xpt, ypt, xsm, ysm):
        dist = []
        for a, b in zip(xsm, ysm):
            d = np.sqrt((xpt - a)**2 + (ypt - b)**2)
            dist.append(d)
        return dist
    
    def map_theta(idx):
        dist = dist_loop(x[idx], y[idx], xsm, ysm )
        sidx=np.argmin(dist)
        arclength = np.sum(np.sqrt(np.diff(xsm[:sidx])**2 + np.diff(ysm[:sidx])**2))
        return round(1 - arclength / arclength_tot, 4)
    
    theta_init = map(map_theta, np.arange(0, len(x)))
    return np.asarray(list(theta_init))


def round_up_to_sig(x, sig=4):
    if x == 0:
        return 0
    else:
        e = floor(log10(abs(x)))
        factor = 10**(sig - 1 - e)
        return ceil(x * factor) / factor
    
