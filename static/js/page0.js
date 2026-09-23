import { firebaseConfig } from './config.js';

const firebaseApp = firebase.initializeApp(firebaseConfig);
const auth = firebase.auth();
const firestore = firebase.firestore();

const signupForm = document.querySelector('.registration.form');
const loginForm = document.querySelector('.login.form');
const forgotForm = document.querySelector('.forgot.form');
const signupBtn = document.querySelector('.signupbtn');
const loginBtn = document.querySelector('.loginbtn');
const forgotBtn = document.querySelector('.forgotbtn');
const anchors = document.querySelectorAll('a');

// Form switching logic
anchors.forEach(anchor => {
  anchor.addEventListener('click', () => {
    const id = anchor.id;
    switch(id) {
      case 'loginLabel':
        signupForm.style.display = 'none';
        loginForm.style.display = 'block';
        forgotForm.style.display = 'none';
        break;
      case 'signupLabel':
        signupForm.style.display = 'block';
        loginForm.style.display = 'none';
        forgotForm.style.display = 'none';
        break;
      case 'forgotLabel':
        signupForm.style.display = 'none';
        loginForm.style.display = 'none';
        forgotForm.style.display = 'block';
        break;
    }
  });
});

// Check if user is authenticated and has verified their email
auth.onAuthStateChanged(user => {
  if (user && user.emailVerified) {
    window.location.href = "/upload";  // Redirect to the upload page if the user is logged in and verified
  }
});

// Signup logic
signupBtn.addEventListener('click', () => {
  const name = document.querySelector('#name').value;
  const username = document.querySelector('#username').value;
  const email = document.querySelector('#email').value.trim();
  const password = document.querySelector('#password').value;

  auth.createUserWithEmailAndPassword(email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      const uid = user.uid;

      user.sendEmailVerification()
        .then(() => {
          alert('Verification email sent. Please check your inbox and verify your email before signing in.');
        })
        .catch((error) => {
          alert('Error sending verification email: ' + error.message);
        });

      firestore.collection('users').doc(uid).set({
        name: name,
        username: username,
        email: email,
      });

      signupForm.style.display = 'none';
      loginForm.style.display = 'block';
      forgotForm.style.display = 'none';
    })
    .catch((error) => {
      alert('Error signing up: ' + error.message);
    });
});

// Login logic
loginBtn.addEventListener('click', () => {
  const email = document.querySelector('#inUsr').value.trim();
  const password = document.querySelector('#inPass').value;

  auth.signInWithEmailAndPassword(email, password)
    .then((userCredential) => {
      const user = userCredential.user;
      if (user.emailVerified) {
        window.location.href = "/upload";  // Redirect to the upload page after login
      } else {
        alert('Please verify your email before logging in.');
      }
    })
    .catch((error) => {
      alert('Error signing in: ' + error.message);
    });
});

// Forgot password logic
forgotBtn.addEventListener('click', () => {
  const emailForReset = document.querySelector('#forgotinp').value.trim();
  if (emailForReset.length > 0) {
    auth.sendPasswordResetEmail(emailForReset)
      .then(() => {
        alert('Password reset email sent. Please check your inbox to reset your password.');
        signupForm.style.display = 'none';
        loginForm.style.display = 'block';
        forgotForm.style.display = 'none';
      })
      .catch((error) => {
        alert('Error sending password reset email: ' + error.message);
      });
  }
});
