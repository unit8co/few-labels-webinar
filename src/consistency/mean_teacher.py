import torch


class MeanTeacher:
    '''
    Basic implementation of mean teacher with exponential moving
    average of the weghts depending on alpha parameter. Scheduler
    can be used to change alpha value during training. Model set
    to eval mode.
    '''
    def __init__(self, alpha, initial_model, scheduler=None):
        self.alpha = alpha
        self.model = initial_model
        self.scheduler = scheduler
        self.model.eval()

    def optimise(self, student_model, epoch):
        if self.scheduler:
            self.alpha = self.scheduler(epoch)
        with torch.no_grad():
            for params, student_params in zip(self.model.parameters(), student_model.parameters()):
                new_params = self.alpha * params + (1- self.alpha) * student_params
                params.copy_(new_params)
        self.model.zero_grad()
        self.model.eval() # not sure needed

