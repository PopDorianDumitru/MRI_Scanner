const awsConfig = {
    Auth: {
        Cognito: {
            userPoolClientId: '5m1pkk61etohe3h0ut17bkuv00',
            userPoolId: 'eu-north-1_woisleA2C',
            loginWith: { // Optional
                oauth: {
                    domain: 'eu-north-1woislea2c.auth.eu-north-1.amazoncognito.com',
                    scopes: ['openid', 'email', 'phone', 'profile'],
                    redirectSignIn: ['http://localhost:5173/', 'https://mri-ai-tool.click/'],
                    redirectSignOut: ['http://localhost:5173/', 'https://mri-ai-tool.click/'],
                    responseType: 'code',
                }
            }
        }
    }
};

export default awsConfig;
