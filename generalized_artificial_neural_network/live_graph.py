import matplotlib.pyplot as PLT

class LiveGraph():

    def __init__(self, graph_title, x_title="", y_title="", epochs=5):
        self.figure = PLT.figure()
        self.figure.suptitle(graph_title)
        PLT.xlabel(x_title)
        PLT.ylabel(y_title)
        PLT.xlim([0, epochs])
        PLT.ylim([0, 1])
        PLT.grid()
        self.error_graph = None
        self.validation_graph = None
        self.figure.show()

    def update(self, error_history, validation_history):
        # PLT.ion()
        if error_history:
            self.error_graph = self.plot(error_history, self.error_graph)
        if validation_history:
            self.validation_graph = self.plot(validation_history, self.validation_graph, invert=True)

        self.figure.canvas.draw()
        PLT.pause(0.00025)
        # PLT.ioff()

    def plot(self, histogram, graph, invert=False):
        yl, xl = [], []
        for x, y in histogram:
            xl.append(x)
            yl.append(y if not invert else 1-y)
        if graph:
            graph.set_xdata(xl)
            graph.set_ydata(yl)
        else: graph = PLT.plot(xl, yl)[0]
        return graph
