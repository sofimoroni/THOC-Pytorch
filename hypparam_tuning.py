import optuna
from exp import THOCTrainer  # Import THOCTrainer to use during optimization
from torch.utils.data import DataLoader

class HyperparameterTuner:
    def __init__(self, train_data, test_data, args, logger, train_loader, test_loader):
        self.train_data = train_data
        self.test_data = test_data
        self.args = args
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader

    def objective(self, trial):
        # Define hyperparameter search space
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)  # Log uniform distribution for learning rate
        hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])  # Number of hidden units
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])  # Batch size
        LAMBDA_orth = trial.suggest_uniform('LAMBDA_orth', 0.1, 1.0)  # Orthogonality loss coefficient
        LAMBDA_TSS = trial.suggest_uniform('LAMBDA_TSS', 0.1, 1.0)  # Temporal self-supervision loss coefficient
        L2_reg = trial.suggest_loguniform('L2_reg', 1e-5, 1e-1)  # L2 regularization coefficient, using log-uniform distribution

        # Set up DataLoader with batch size
        train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)

        # Initialize the model with the trial parameters
        self.args.hidden_dim = hidden_dim
        self.args.LAMBDA_orth = LAMBDA_orth
        self.args.LAMBDA_TSS = LAMBDA_TSS
        self.args.L2_reg = L2_reg

        # Initialize the trainer with the updated args
        trainer = THOCTrainer(
            args=self.args,
            logger=self.logger,
            train_loader=train_loader,
            test_loader=test_loader,
        )

        # Train the model
        trainer.train()

        # Evaluate the model on the test set and get the validation loss
        validation_loss = trainer.infer()['F1']  # This assumes the test method returns validation loss

        trial.report(validation_loss, step=1)  # Report the validation loss to Optuna

        # Handle pruning if necessary
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return validation_loss  # Minimize the validation loss

    def tune(self, n_trials=50):
        # Create an Optuna study to minimize validation loss
        study = optuna.create_study(direction='maximize')  # 'minimize' for validation loss
        study.optimize(self.objective, n_trials=n_trials)  # Perform n_trials optimization

        # Get the best hyperparameters from the study
        best_params = study.best_params
        self.logger.info(f"Best Hyperparameters: {best_params}")
        return best_params

