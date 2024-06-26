open Lwt.Infix
open Cohttp
open Cohttp_lwt_unix

let fetch url =
  Client.get (Uri.of_string url) >>= fun (resp, body) ->
  let code = resp |> Response.status |> Code.code_of_status in
  let headers = resp |> Response.headers |> Header.to_string in
  body |> Cohttp_lwt.Body.to_string >|= fun body ->
  Printf.printf "Response code: %d\nHeaders:\n%s\nBody:\n%s\n" code headers body

let () =
  let url = "https://www.example.com" in
  Lwt_main.run (fetch url)
