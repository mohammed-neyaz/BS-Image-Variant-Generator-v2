export function setupImagePreview() {
    const previewSection = document.getElementById('previewSection');
    const uploadSection = document.getElementById('uploadSection');
    const changeImageBtn = document.getElementById('change-image-btn');
    const variantsSection = document.getElementById('variantsSection');
    const variantsGrid = document.getElementById('variantsGrid');
    const promptSection = document.getElementById('promptSection');

    document.addEventListener('imageSelected', async (e) => {
        const file = e.detail.file;
        
        previewSection.classList.remove('hidden');
        uploadSection.classList.add('hidden');
        
        const reader = new FileReader();
        reader.onload = function(e) {
            imagePreview.src = e.target.result;
        };
        reader.readAsDataURL(file);

        const event = new CustomEvent('generatePrompt', { detail: { file } });
        document.dispatchEvent(event);
    });
    // Button to upload a new image
    changeImageBtn.onclick = async () =>{
        // Reset the input and preview
        previewSection.classList.add('hidden');
        changeImageBtn.classList.add('hidden'); // Hide the change image button
        variantsGrid.innerHTML = '';
        variantsSection.classList.add('hidden');
        promptSection.classList.add('hidden');
        uploadSection.classList.remove('hidden'); // Show drag-drop section again
    };
}