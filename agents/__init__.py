"""Policy, generative, and data-collection agents."""

from agents.chunk_tdflow import ChunkTDFlowModel
from agents.crl import CRLAgent
from agents.gcbc import GCBCAgent
from agents.gcfbc import GCFBCAgent
from agents.gciql import GCIQLAgent
from agents.hiql import HIQLAgent
from agents.onestep import OneStepFlowModel
from agents.qrl import QRLAgent
from agents.sac import SACAgent
from agents.tdflow import FlowModel
from agents.vae import VAE

agents = dict(
    chunk_tdflow=ChunkTDFlowModel,
    crl=CRLAgent,
    gcbc=GCBCAgent,
    gcfbc=GCFBCAgent,
    gciql=GCIQLAgent,
    hiql=HIQLAgent,
    onestep=OneStepFlowModel,
    qrl=QRLAgent,
    sac=SACAgent,
    tdflow=FlowModel,
    vae=VAE,
)
