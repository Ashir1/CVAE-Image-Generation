class Encoder(nn.Module):
    def __init__(self, latent_dim, text_embedding_dim, concat_input_and_condition=True):
        super(Encoder, self).__init__()
        self.use_cond_input = concat_input_and_condition
        self.enc_block_1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)  # 64 -> 32
        self.enc_block_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 32 -> 16
        self.enc_block_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)  # 16 -> 8
        self.enc_block_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  # 8 -> 4
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(256 * 4 * 4 + text_embedding_dim, latent_dim * 2) 

    def forward(self, input_img, text_embedding, is_train):
        x = F.leaky_relu(self.enc_block_1(input_img))
        x = F.batch_norm(x, training=is_train)
        x = F.leaky_relu(self.enc_block_2(x))
        x = F.batch_norm(x, training=is_train)
        x = F.leaky_relu(self.enc_block_3(x))
        x = F.batch_norm(x, training=is_train)
        x = F.leaky_relu(self.enc_block_4(x))
        x = F.batch_norm(x, training=is_train)

        x = self.flatten(x)
        # Concatenate flattened image features with text embedding
        x = torch.cat([x, text_embedding], dim=1)  
        x = self.dense(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, batch_size=32):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dense = nn.Linear(latent_dim + label_dim, 4 * 4 * 256)  
        self.dec_block_1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)  # 4 -> 8
        self.dec_block_2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)  # 8 -> 16
        self.dec_block_3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)   # 16 -> 32
        self.dec_block_4 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)   # 32 -> 64

    def forward(self, z_cond, is_train):
        x = F.leaky_relu(self.dense(z_cond))
        x = x.view(-1, 256, 4, 4) 
        x = F.leaky_relu(F.batch_norm(self.dec_block_1(x), training=is_train))
        x = F.leaky_relu(F.batch_norm(self.dec_block_2(x), training=is_train))
        x = F.leaky_relu(F.batch_norm(self.dec_block_3(x), training=is_train))
        x = self.dec_block_4(x)  
        return x

class ConvCVAE(nn.Module):
    def __init__(self, encoder, decoder, text_vocab_size, text_embedding_dim, latent_dim, beta=1, image_dim=(64, 64, 3)):
        super(ConvCVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.text_embedding_dim = text_embedding_dim
        self.beta = beta
        self.image_dim = image_dim

        # Text embedding layer
        self.text_embedding = nn.Embedding(text_vocab_size, text_embedding_dim)

    def forward(self, inputs, is_train):
        input_img, captions = inputs
        # Compute text embeddings from captions
        text_embedding = torch.mean(self.text_embedding(captions), dim=1)  
        
        enc_output = self.encoder(input_img, text_embedding, is_train)
        z_mean, z_log_var = torch.chunk(enc_output, 2, dim=1)
        z = self.reparametrization(z_mean, z_log_var)
        logits = self.decoder(z, is_train)
        recon_img = torch.sigmoid(logits)
        
        # Loss 
        latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1)
        reconstr_loss = F.binary_cross_entropy(recon_img.view(recon_img.size(0), -1), input_img.view(input_img.size(0), -1), reduction='sum')
        loss = reconstr_loss + self.beta * latent_loss.mean()
        return {
            'recon_img': recon_img,
            'latent_loss': latent_loss,
            'reconstr_loss': reconstr_loss,
            'loss': loss,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }

    def reparametrization(self, z_mean, z_log_var):
        eps = torch.randn(z_mean.size(0), self.latent_dim).to(z_mean.device)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps
        return z