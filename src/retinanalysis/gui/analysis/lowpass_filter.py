"""Low-pass Butterworth filter plugin."""
 
import param
import numpy as np
from retinanalysis.gui.analysis._base import AnalysisPlugin
 
 
class LowPassFilter(AnalysisPlugin):
    name = param.String(default="Low-Pass Filter")
    description = param.String(default="Butterworth low-pass filter applied with zero-phase filtfilt")
    cutoff_hz = param.Number(default=500, bounds=(1, 10000), doc="Cutoff frequency in Hz")
    order = param.Integer(default=4, bounds=(1, 10), doc="Filter order")
 
    def process(self, trace, sample_rate, **kwargs):
        from scipy.signal import butter, filtfilt
        nyq = sample_rate / 2.0
        if self.cutoff_hz >= nyq:
            return trace
        b, a = butter(self.order, self.cutoff_hz / nyq, btype='low')
        return filtfilt(b, a, trace)
 
    def get_label(self):
        return f"LP {self.cutoff_hz} Hz"