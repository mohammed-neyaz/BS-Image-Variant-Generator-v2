export function setupVariantGeneration() {
    const promptSection = document.getElementById('promptSection');
    const generatedPrompt = document.getElementById('generatedPrompt');
    const variantsSection = document.getElementById('variantsSection');
    const variantsGrid = document.getElementById('variantsGrid');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const changeImageBtn = document.getElementById('change-image-btn');

    const loginDiv = document.getElementById('loginDiv');
const contentDiv = document.getElementById('contentDiv');

    const username = document.getElementById('username');
    const password = document.getElementById('password');

    let prompt = '';
    let imgHeight = 512;
    let imgWidth = 512;
    let variant = 0;

    document.addEventListener('generatePrompt', async (e) => {
        try {
            showLoading();
            variant = 1;
            const file = e.detail.file;
            
            const formData = new FormData();
            formData.append('image', file);

            const promptResponse = await fetch('/generate-prompt', {
                method: 'POST',
                body: formData
            });

            if(promptResponse.status === 401){
                loginDiv.classList.remove('hidden');
                contentDiv.classList.add('hidden');
                username.value = '';
                password.value = '';
                alert('Your session has expired. Please login again.');
                return;
            }

            if (!promptResponse.ok) throw new Error('Failed to generate prompt');
            
            const promptData = await promptResponse.json();
            
            promptSection.classList.remove('hidden');
            generatedPrompt.textContent = promptData.prompt;
            prompt = promptData.prompt;

            const img = new Image();
            img.src = URL.createObjectURL(file);
            await new Promise((resolve) => {
                img.onload = resolve;
            });
            imgHeight = img.height;
            imgWidth = img.width;

            const variantsResponse = await fetch('/generate-variants', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    prompt: prompt,
                    width: imgWidth,
                    height: imgHeight
                })
            });

            if(promptResponse.status === 401){
                loginDiv.classList.remove('hidden');
                contentDiv.classList.add('hidden');
                username.value = '';
                password.value = '';
                alert('Your session has expired. Please login again.');
                return;
            }

            if (!variantsResponse.ok) throw new Error('Failed to generate variants');
            
            const variantsData = await variantsResponse.json();
            
            displayVariants(variantsData.images);
            
            // Dispatch event to stop timer
            document.dispatchEvent(new CustomEvent('variantsGenerated'));
            
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing your image. Please try again.');
        } finally {
            hideLoading();
        }
    });

    function displayVariants(images) {
        variantsSection.classList.remove('hidden');
        

        // Create a promise to preload all images
        const preloadImages = images.map(image => {
            return new Promise((resolve) => {
                const img = new Image();
                img.src = image.url;
                img.onload = () => resolve(image); // Resolve with the image once loaded
            });
        });

        // Wait for all images to preload
        Promise.all(preloadImages).then(loadedImages => {
            // Display the generated variants
            loadedImages.forEach(image => {
                const variantContainer = document.createElement('div');
                variantContainer.className = 'bg-stone-800 p-4 rounded-lg shadow-md';
                
                const img = document.createElement('img');
                img.src = image.url;
                img.alt = 'Generated variant';
                img.className = 'w-full h-auto rounded';
                
                const downloadBtn = document.createElement('a');
                downloadBtn.href = image.url;
                downloadBtn.download = 'variant-' + variant + '.webp';
                downloadBtn.className = 'mt-2 inline-block bg-lime-400 text-stone-900 px-4 py-2 rounded hover:bg-lime-600 transition-colors text-center w-full';
                downloadBtn.textContent = 'Download';
                
                variantContainer.appendChild(img);
                variantContainer.appendChild(downloadBtn);
                variantsGrid.appendChild(variantContainer);
                variant++;
            });

            // Show the button to generate more variants
            const generateMoreBtn = document.getElementById('generateMoreBtn');
            generateMoreBtn.classList.remove('hidden'); // Show the button

            // Add event listener for the button
            generateMoreBtn.onclick = async () => {
                document.dispatchEvent(new CustomEvent('generateVariants'));
                showLoading(); // Show loading indicator
                generateMoreBtn.classList.add('hidden'); // Hide the button while generating

                try {
                    
                    const variantsResponse = await fetch('/generate-variants', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            prompt: prompt,
                            width: imgWidth,
                            height: imgHeight
                        })
                    });

                    if (!variantsResponse.ok) throw new Error('Failed to generate more variants');
                    
                    const variantsData = await variantsResponse.json();
                    displayVariants(variantsData.images); // Call the display function again
                    // Dispatch event to stop timer
            document.dispatchEvent(new CustomEvent('variantsGenerated'));
            
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred while generating more variants. Please try again.');
                } finally {
                    hideLoading(); // Hide loading indicator
                    generateMoreBtn.classList.remove('hidden'); // Show the button again
                }
            };

            changeImageBtn.classList.remove('hidden');
        });
    }

    function showLoading() {
        loadingIndicator.classList.remove('hidden');
    }

    function hideLoading() {
        loadingIndicator.classList.add('hidden');
    }
}