from .gates_model import ImprovedGATES

def create_enhanced_model(config, in_channels):
    """创建增强版GATES模型"""
    return ImprovedGATES(
        in_channels=in_channels,
        hidden_channels=config['model']['hidden_dims'][0],
        out_channels=config['model']['hidden_dims'][1],
        alpha=config['model']['alpha'],
        spatial_att_heads=config['model'].get('spatial_att_heads', 4),
        gene_att_heads=config['model'].get('gene_att_heads', 4)
    )

def get_trainer_class():
    """返回增强版训练器类"""
    from .trainer import ImprovedGATESTrainer
    return ImprovedGATESTrainer
