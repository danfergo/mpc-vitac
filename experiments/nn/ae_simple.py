from torch import nn
import torch


class AutoEncoderSimple(nn.Module):
    """
        Simpler e2e model give, vision_t, touch_t, action_t
        predicts the vision_t, touch_t and action_t at the next timestep
    """

    def __init__(self,
                 vis_ae=None,
                 touch_ae=None,
                 ):
        super(AutoEncoderSimple, self).__init__()
        self.vis_ae = vis_ae
        self.touch_ae = touch_ae
        # self.associative_cortex = parietal_cortex

    def forward(self, v_t0, l_t0):
        # encode/project observations
        ev_t0 = self.vis_ae.encode(v_t0)
        el_t0 = self.touch_ae.encode(l_t0)
        # er_t0 = self.touch_cortex.encode(r_t0)

        # em_t0 = # ??? \
        # self.touch_cortex.encode(m_t0)
        # em_t0 = m_t0[:, :, None, None]
        # e_size = ev_t0.size()[2]
        # em_t0 = torch.tile(em_t0, (1, 1, e_size, e_size))

        # associate vision and touch (s_c = current state)
        es_t0 = torch.cat((ev_t0, el_t0), dim=1)  # self.associative_cortex.associate(ev_t0, el_t0, er_t0, em_t0)
        es_tx = es_t0
        # predict next state (s_n = next state)
        # es_t1 = self.temporal_cortex.forward(es_t0)

        # ev_t1, el_t1, er_t1, m_t1 = self.associative_cortex.dissociate(es_t1)
        # dissociate vision and touch (s_c = current state)
        m = es_tx.shape[1] // 2  # middle point, in the channels dimension
        ev_tx, el_tx = es_tx[:, :m, :, :], es_tx[:, m:, :, :]

        # decode (next) observations
        v_tx = self.vis_ae.decode(ev_tx)
        l_tx = self.touch_ae.decode(el_tx)
        # r_t1 = self.touch_cortex.decode(er_t1)
        # m_t1 = self.motor_cortex.decode(er_t1)

        return v_tx, l_tx
