import torch.onnx 

from vqvae import VQVAE_Decoder ,VQVAE_Encoder, VQVAE_all




#Function to Convert to ONNX 
def Convert_ONNX(): 

    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn((1, 3, 32, 32))  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         "vqvae.onnx",       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
         opset_version=10,    # the ONNX version to export the model to 
         do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                                'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX')

if __name__ == "__main__": 


    # Load the model from the file
    batch_size = 64
    image_size = 32
    hidden_dim = 128
    residual_hidden_dim = 32
    num_residual_lyers = 2
    embedding_dim = 32
    num_embedding = 128 #K
    commitence_cost = 0.25
    use_ema = False
    ema_decay = 0.95
    lr = 1e-4
    epochs = 50

    encoder = VQVAE_Encoder(in_channels = 3, hidden_channels = hidden_dim, residual_hiddens_num = num_residual_lyers, residual_hidden_dim = residual_hidden_dim)
    decoder = VQVAE_Decoder(in_channels = embedding_dim, hidden_channels = hidden_dim, residual_hiddens_num = num_residual_lyers, residual_hidden_dim = residual_hidden_dim)
    model = VQVAE_all(encoder = encoder, decoder = decoder, hidden_dim = hidden_dim, embedding_dim = embedding_dim,num_embeddings = num_embedding ,commitment_cost =commitence_cost, data_variance = 1.0)


    model_file_path = "./vqvae_model.pth"
    model.load_state_dict(torch.load(model_file_path)) 

    Convert_ONNX()