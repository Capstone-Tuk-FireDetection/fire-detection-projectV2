importScripts('https://www.gstatic.com/firebasejs/10.11.0/firebase-app-compat.js');
importScripts('https://www.gstatic.com/firebasejs/10.11.0/firebase-messaging-compat.js');

firebase.initializeApp({
    apiKey: 'AIzaSyBai7wOP8l_Sia-SWyiXvqZmwnL0YWCwfE',
    appId: '1:33909260846:web:1b916db809b3bba76055bf',
    messagingSenderId: '33909260846',
    projectId: 'firedetection-3d3d5',
    authDomain: 'firedetection-3d3d5.firebaseapp.com',
    storageBucket: 'firedetection-3d3d5.firebasestorage.app',
    measurementId: 'G-9XL3EHPDZR',
});
const messaging = firebase.messaging();