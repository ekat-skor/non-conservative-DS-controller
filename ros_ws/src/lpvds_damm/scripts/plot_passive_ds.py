#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PassiveDSPlotter:
    def __init__(self):
        rospy.init_node('passive_ds_plotter')
        # storage for time-series
        self.t = []
        self.alpha = []
        self.br = []
        self.bs = []
        self.s  = []
        self.sdot = []
        self.z   = []
        self.latest = {'alpha':0,'br':0,'bs':0,'s':0,'sdot':0,'z':0}

        # subscribers
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/alpha_',          Float32, lambda m: self._cb('alpha', m))
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/beta_r_',         Float32, lambda m: self._cb('br',    m))
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/beta_s_',         Float32, lambda m: self._cb('bs',    m))
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/s_',  Float32, lambda m: self._cb('s',     m))
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/sdot_',           Float32, lambda m: self._cb('sdot',  m))
        rospy.Subscriber('/nc_passive_ds_impedance_controller/passive_ds/z_',        Float32, lambda m: self._cb('z',     m))

        # timer to sample at fixed rate
        self.start = rospy.Time.now()
        rospy.Timer(rospy.Duration(0.02), self._sample)  # 50 Hz

        # set up plots
        self.fig, axes = plt.subplots(6,1, sharex=True, figsize=(8,10))
        titles = ['alpha','beta_r','beta_s','s','sdot','z']
        for ax,t in zip(axes,titles):
            ax.set_ylabel(t)
        axes[-1].set_xlabel('time [s]')
        self.lines = [ax.plot([],[])[0] for ax in axes]

    def _cb(self, key, msg):
        self.latest[key] = msg.data

    def _sample(self, event):
        t = (event.current_real - self.start).to_sec()
        self.t.append(t)
        self.alpha.append(self.latest['alpha'])
        self.br.append(self.latest['br'])
        self.bs.append(self.latest['bs'])
        self.s.append(self.latest['s'])
        self.sdot.append(self.latest['sdot'])
        self.z.append(self.latest['z'])

    def animate(self, i):
        data = [self.alpha, self.br, self.bs, self.s, self.sdot, self.z]
        for line, y in zip(self.lines, data):
            line.set_data(self.t, y)
        # keep last 10 s on screen
        if self.t:
            tmin = max(0, self.t[-1] - 10)
            for ax in self.fig.axes:
                ax.set_xlim(tmin, self.t[-1])
                ax.relim(); ax.autoscale_view(True,True,True)
        return self.lines

    def run(self):
        ani = FuncAnimation(self.fig, self.animate, interval=100)
        plt.tight_layout()
        plt.show()
        rospy.spin()

if __name__ == '__main__':
    PassiveDSPlotter().run()
