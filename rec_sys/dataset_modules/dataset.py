from torch_geometric.loader import DataLoader


class HeteroDataLoader(DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=8,
        template=None,
        **kwargs,
    ):
        """
        Наследуемся от DataLoader PyG и добавляем проверку структуры.
        template: шаблон для проверки структуры HeteroData
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            **kwargs,
        )
        self.template = template

    def __iter__(self):
        for batch in super().__iter__():
            if self.template is not None:
                for graph in batch.to_data_list():
                    if not self._check_structure(graph):
                        raise ValueError("Graph does not match template")
            yield batch

    def _check_structure(self, graph):
        """Простейшая проверка структуры по шаблону"""
        for key, sub_template in self.template.items():
            if key not in graph:
                print(f"Missing key: {key}")
                return False
            if isinstance(sub_template, dict):
                sub_graph = graph[key]
                for sub_key in sub_template:
                    if sub_key not in sub_graph:
                        print(f"Missing sub-key: {sub_key} in {key}")
                        return False
        return True
