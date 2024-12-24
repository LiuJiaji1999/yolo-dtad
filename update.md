

1. 加载数据，双数据输入，source-train，target-train
    uda_trainer.py:
        self.targetset = self.get_dataset(self.data)

        self.target_loader = self.get_dataloader(self.targetset, batch_size=batch_size, rank=RANK, mode="target")

    import albumentations as A
    from utils.daca import get_best_region, transform_img_bboxes


