# Rank Predictor

The rank predictor Python module contains ML model definitions, training scripts, and the data pipeline for webpage rank prediction.
It uses the PyTorch framework.

## Sacred

### Monitoring

```bash
sudo service mongod start
sacredboard -mu mongodb://localhost:27017/sacred sacred
```