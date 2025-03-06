import { setupDropZone } from './dropZone.js';
import { setupImagePreview } from './imagePreview.js';
import { setupVariantGeneration } from './variantGeneration.js';
import { setupTimer } from './timer.js';

document.addEventListener('DOMContentLoaded', () => {
    setupDropZone();
    setupImagePreview();
    setupVariantGeneration();
    setupTimer();
});

// Define the login and content divs
const loginDiv = document.getElementById('loginDiv');
const contentDiv = document.getElementById('contentDiv');
const loginForm = document.getElementById('loginForm');

// Show the login div and hide the content div by default
loginDiv.classList.remove('hidden');
contentDiv.classList.add('hidden');

// Handle login form submission
loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;

    // Send login request to the backend
    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: {
                'Authorization': 'Basic ' + btoa(username + ':' + password),
                'Content-Type': 'application/json',
            },
        });

        if (response.ok) {
            // Hide login div and show content div on successful login
            loginDiv.classList.add('hidden');
            contentDiv.classList.remove('hidden');
        } else {
            alert('Login failed. Please check your username and password.');
        }
    } catch (error) {
        console.error('Error during login:', error);
        alert('An error occurred during login. Please try again.');
    }
});