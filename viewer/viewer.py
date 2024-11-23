import viser

from viewer.client import ClientThread


class Viewer:
    def __init__(self, gaussians) -> None:
        self.gaussians = gaussians
        self.clients = {}

        host = "0.0.0.0"
        port = 8080
        self.server = viser.ViserServer(host=host, port=port)
        print(f"http://localhost:{port}")

        self.server.gui.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        self.server.on_client_connect(self._handle_new_client)
        self.server.on_client_disconnect(self._handle_disconnect)

    def _handle_new_client(self, client: viser.ClientHandle):
        client_thread = ClientThread(self, self.gaussians, client)
        client_thread.start()
        self.clients[client.client_id] = client_thread

    def _handle_disconnect(self, client: viser.ClientHandle):
        try:
            self.clients[client.client_id].stop()
            del self.clients[client.client_id]
        except Exception as e:
            print(e)

    def update_scene(self, gaussians):
        self.gaussians = gaussians
        for _, client in self.clients.items():
            client.gaussians = gaussians
