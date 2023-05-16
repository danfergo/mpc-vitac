from torch import nn
import torch


class SimplerBrain(nn.Module):
    """
        Simpler e2e model give, vision_t, touch_t, action_t
        predicts the vision_t, touch_t and action_t at the next timestep
    """

    def __init__(self,
                 vis_cortex=None,
                 touch_cortex=None,
                 temporal_cortex=None
                 ):
        super(SimplerBrain, self).__init__()
        self.vis_cortex = vis_cortex
        self.touch_cortex = touch_cortex
        # self.associative_cortex = parietal_cortex
        self.temporal_cortex = temporal_cortex

    def forward(self, v_t0, l_t0, m_t0):
        # encode/project observations
        ev_t0 = self.vis_cortex.encode(v_t0)
        el_t0 = self.touch_cortex.encode(l_t0)
        # er_t0 = self.touch_cortex.encode(r_t0)

        # em_t0 = # ??? \
        # self.touch_cortex.encode(m_t0)
        em_t0 = m_t0[:, :, None, None]
        e_size = ev_t0.size()[2]
        em_t0 = torch.tile(em_t0, (1, 1, e_size, e_size))

        # associate vision and touch (s_c = current state)
        es_t0 = torch.cat((ev_t0, el_t0, em_t0), dim=1)  # self.associative_cortex.associate(ev_t0, el_t0, er_t0, em_t0)

        # predict next state (s_n = next state)
        es_t1 = self.temporal_cortex.forward(es_t0)
        # dissociate vision and touch (s_c = current state)

        # ev_t1, el_t1, er_t1, m_t1 = self.associative_cortex.dissociate(es_t1)
        m = es_t1.shape[1] // 2  # middle point, in the channels dimension
        ev_t1, el_t1 = es_t1[:, :m, :, :], es_t1[:, m:, :, :]

        # decode (next) observations
        v_t1 = self.vis_cortex.decode(ev_t1)
        l_t1 = self.touch_cortex.decode(el_t1)
        # r_t1 = self.touch_cortex.decode(er_t1)
        # m_t1 = self.motor_cortex.decode(er_t1)

        return v_t1, l_t1
