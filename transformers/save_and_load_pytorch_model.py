model = MyModel(num_classes).to(device)
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
output_model = './models/model_xlnet_mid.pth'

# save
def save(model, optimizer):
    # save
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)

save(model, optimizer)

# load
checkpoint = torch.load(output_model, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])



#############transformers load pre-trainned model
model = ClassificationModel('bert', 'outputs/checkpoint-679-epoch-1', num_labels=5)
