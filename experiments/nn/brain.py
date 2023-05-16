from torch import nn


class Brain(nn.Module):
    """
        Main model give, vision_t, touch_t, action_t
        predicts the vision_t, touch_t and action_t at the next timestep
    """

    def __init__(self,
                 vis_cortex=None,
                 touch_cortex=None,
                 parietal_cortex=None,
                 temporal_cortex=None
                 ):
        super(Brain, self).__init__()
        self.vis_cortex = vis_cortex
        self.touch_cortex = touch_cortex
        self.parietal_cortex = parietal_cortex
        self.temporal_cortex = temporal_cortex

    def forward(self, v_t0, l_t0, r_t0, m_t0):
        # encode/project observations
        ev_t0 = self.vis_cortex.encode(v_t0)
        el_t0 = self.touch_cortex.encode(l_t0)
        er_t0 = self.touch_cortex.encode(r_t0)
        em_t0 = self.touch_cortex.encode(m_t0)

        # associate vision and touch (s_c = current state)
        es_t0 = self.parietal_cortex.associate(ev_t0, el_t0, er_t0, em_t0)

        # predict next state (s_n = next state)
        es_t1 = self.temporal_cortex.next(es_t0)

        # dissociate vision and touch (s_c = current state)
        ev_t1, el_t1, er_t1, m_t1 = self.parietal_cortex.dissociate(es_t1)

        # decode (next) observations
        v_t1 = self.vis_cortex.decode(ev_t1)
        l_t1 = self.touch_cortex.decode(el_t1)
        r_t1 = self.touch_cortex.decode(er_t1)
        m_t1 = self.motor_cortex.decode(er_t1)

        return v_t1, l_t1, r_t1, m_t1
