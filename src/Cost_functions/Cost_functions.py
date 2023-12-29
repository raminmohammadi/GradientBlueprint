class CostFunction:
    @staticmethod
    def log_loss(y_hat, y):
        if y[0] == 1:
            loss = y_hat.log()
        else:
            loss = (1 - y_hat).log()
        return -loss
    
    @staticmethod    
    def categorical_cross_entropy_loss(y_hat, y):
        loss = y_hat[y[0]].log()
        return -loss
    
    @staticmethod
    def sse(y_hat, y):
        loss = (y_hat - y)**2
        return loss