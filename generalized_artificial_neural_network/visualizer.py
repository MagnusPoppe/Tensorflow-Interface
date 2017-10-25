class VisualizationAdapter():

    def __init__(self, trainer):
        self.hinton_figures = []     # list of matplotlib.pyplot.Figure
        self.dendrogram_figures = [] # list of matplotlib.pyplot.Figure
        self.matrix_figures = [] # list of matplotlib.pyplot.Figure
        self.trainer = trainer

    def display_hinton_graph_from_training_history(self):
        self._initialize_graphics()
        from downing_code.tflowtools import hinton_plot
        for history in self.trainer.monitored_modules_history:
            self._draw_hinton_graph(history[1], history[0], hinton_plot)

    def _draw_hinton_graph(self, graph_results, epoch, modules, imported_method=None):
        if not imported_method: from downing_code.tflowtools import hinton_plot
        else: hinton_plot = imported_method
        self._initialize_graphics(modules)
        if self.hinton_figures and graph_results:
            # Local import to be able to run on server.
            for i in range(len(modules)):
                hinton_plot(matrix=graph_results[i],
                            fig=self.hinton_figures[i],
                            title=modules[i].name + " @ epoch=" + str(epoch))

    def _draw_dendrograms(self, features, labels):
        from downing_code.tflowtools import dendrogram
        import matplotlib.pyplot as PLT
        for i, monitored in enumerate(features):
            module_name = self.trainer.monitored_modules[i].name
            if any(word in module_name for word in ["in", "bias", "out"]):
                self.dendrogram_figures.append(PLT.figure())
                dendrogram(features=monitored, labels=labels, figure=self.dendrogram_figures[-1], title=module_name)

    def _initialize_graphics(self, modules=None):
        """ Creates the list of "matplotlib.pyplot.Figure" to be used with the hinton diagrams. """
        import matplotlib.pyplot as PLT
        for i in range(len(self.trainer.monitored_modules if modules is None else modules)):
            self.hinton_figures.append(PLT.figure())

    def close_all_matplotlib_windows(self):
        import matplotlib.pyplot as PLT
        PLT.close('all')
        self.hinton_figures = []
        self.dendrogram_figures = []
        self.trainer.graph = None

    def display_matrix(self, modules:list, matrixes:list):
        from downing_code.tflowtools import display_matrix
        import matplotlib.pyplot as PLT
        for i, matrix in enumerate(matrixes):
            try: x = matrix.shape[1]-1
            except Exception: matrix.shape=(matrix.shape[0], 1)
            self.matrix_figures += [PLT.figure()]
            display_matrix(matrix, fig=self.matrix_figures[-1], title=modules[i].name)
