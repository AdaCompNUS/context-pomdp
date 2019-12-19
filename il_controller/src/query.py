import torch
import socketio, eventlet
import eventlet.wsgi
from flask import Flask

sio = socketio.Server()
app = Flask(__name__)

from Data_processing import global_params


global_config = global_params.config

model_device_id = global_config.GPU_devices[0]
device = torch.device("cuda:" + str(model_device_id) if torch.cuda.is_available() else "cpu")



@sio.on('ready')
def on_ready(sid, data):
    print("Received data from client %s" % sid)
    data = data.to(device)

    value, acc_pi, acc_mu, acc_sigma, \
    ang, _, _ = drive_net.forward(data)

    return value.cpu(), acc_pi.cpu(), acc_mu.cpu(), acc_sigma.cpu(), ang.cpu()


if __name__ == '__main__':
    drive_net = torch.jit.load("torchscript_version.pt").cuda(device)

    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8080)), app)