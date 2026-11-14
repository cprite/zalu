# ZALU dashboard

Requires Node.js 20.9 or later. Unzip the project and run:

```sh
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and choose **Run sample inference**. The sample adapter runs in the browser with deterministic mock detections. For a production build, run `npm run build` and then `npm start`.

The project includes the generated DashboardSpec at `src/dashboard/dashboard.json`, its shared JSON Schema and the same React renderer used by ZALU. `src/dashboard/sample-data.json` contains the class vocabulary, event target and confidence used by the sample adapter.

To use real data, expand **Data connection** and enter your ZALU backend URL, the ID of a deployed model with the same classes, and a token if required. The live adapter uses multipart `POST /api/inference`. The token is held in memory and entered at runtime. Use HTTPS or localhost for camera access.
